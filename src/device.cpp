#include "device.hpp"

#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

#include <drm.h>
#include <xf86drm.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "sid.h"

Buffer::Buffer(Device& device, std::uint64_t size, unsigned options) : device_{&device}, map_count_{0}
{
	size_ = size = (size + 32767) & ~32767;

	drm_amdgpu_gem_create create_args = { };
	create_args.in.bo_size = size;
	create_args.in.alignment = 4096;
	create_args.in.domains =
			(options & vram) ? AMDGPU_GEM_DOMAIN_VRAM : AMDGPU_GEM_DOMAIN_GTT;
	create_args.in.domain_flags = AMDGPU_GEM_CREATE_CPU_ACCESS_REQUIRED;

	if (drmCommandWriteRead(device.fd, DRM_AMDGPU_GEM_CREATE, &create_args,
			sizeof(create_args))) {
		throw -1;

	}
	handle_ = create_args.out.handle;

	drm_amdgpu_gem_va va_args = { };
	va_args.operation = AMDGPU_VA_OP_MAP;
	va_args.handle = handle_;
	va_args.offset_in_bo = 0;
	va_args.flags = AMDGPU_VM_PAGE_READABLE | AMDGPU_VM_PAGE_WRITEABLE
			| AMDGPU_VM_PAGE_EXECUTABLE;
	va_args.map_size = size;
	gpu_address_ = va_args.va_address = alloc_va(device, size);

	if (int r = drmCommandWriteRead(device.fd, DRM_AMDGPU_GEM_VA, &va_args,
			sizeof(va_args))) {
		drm_gem_close args = { };

		args.handle = handle_;
		drmIoctl(device.fd, DRM_IOCTL_GEM_CLOSE, &args);

		throw -1;
	}
}

Buffer::~Buffer()
{
	drm_amdgpu_gem_va va_args = { };
	va_args.operation = AMDGPU_VA_OP_UNMAP;
	va_args.handle = handle_;
	va_args.offset_in_bo = 0;
	va_args.flags = AMDGPU_VM_PAGE_READABLE | AMDGPU_VM_PAGE_WRITEABLE
			| AMDGPU_VM_PAGE_EXECUTABLE;
	va_args.map_size = size_;
	va_args.va_address = gpu_address_;

	drmCommandWriteRead(device_->fd, DRM_AMDGPU_GEM_VA, &va_args, sizeof(va_args));
	free_va(*device_, gpu_address_, size_);

	drm_gem_close close_args = {};

	close_args.handle = handle_;
	drmIoctl(device_->fd, DRM_IOCTL_GEM_CLOSE, &close_args);
}

void* Buffer::map() {
	if (map_count_) {
		++map_count_;
		return map_pointer_;
	}

	drm_amdgpu_gem_mmap map_args;
	map_args.in.handle = handle_;
	if (drmCommandWriteRead(device_->fd, DRM_AMDGPU_GEM_MMAP, &map_args,
			sizeof(map_args))) {
		throw -1;
	}

	map_pointer_ = mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, device_->fd, map_args.out.addr_ptr);
	if (map_pointer_ == MAP_FAILED)
		throw -1;

	++map_count_;
	return map_pointer_;
}

void Buffer::unmap() {
	if (!--map_count_)
		munmap(map_pointer_, size_);
}


std::uint64_t Buffer::alloc_va(Device& dev, std::uint64_t size)
{
	std::uint64_t base = dev.device_info_.virtual_address_offset;
	for (auto it = dev.va_ranges.begin(); it != dev.va_ranges.end(); ++it) {
		if (it->first - base >= size) {
			dev.va_ranges.insert(it, { base, base + size });
			return base;
		}
		base = it->second;
	}
	if (dev.device_info_.virtual_address_max - base >= size) {
		dev.va_ranges.push_back( { base, base + size });
		return base;
	}
	std::abort();
}

void Buffer::free_va(Device& dev, std::uint64_t address, std::uint64_t size)
{
	for(auto it = dev.va_ranges.begin(); it != dev.va_ranges.end(); ++it) {
		if(it->first == address && it->second == address + size) {
			dev.va_ranges.erase(it);
			return;
		} else if(it->first == address) {
			it->first = address + size;
			return;
		} else if(it->second == address + size) {
			it->second = address;
			return;
		} else if(it->first < address && it->second > address + size) {
			auto old_begin = it->first;
			it->first = address + size;
			dev.va_ranges.insert(it, {old_begin, address});
			return;
		}
	}
	std::abort();
}

void upload(Buffer& buf, void const* data, std::uint64_t offset, std::size_t size) {
	 auto dest = reinterpret_cast<char*>(buf.map());
	 std::memcpy(dest + offset, data, size);
	 buf.unmap();
}

Command_buffer::Command_buffer(Device& dev, unsigned hw_type, unsigned queue) : device_{&dev}, hw_type_{hw_type}, queue_{queue}, submit_handle_{~0ull} {}


namespace {
class Buffer_list {
public:
	template<typename It>
	Buffer_list(int fd, It b, It e);
	~Buffer_list();

	Buffer_list(Buffer_list const&) = delete;
	Buffer_list& operator=(Buffer_list&) = delete;

	std::uint32_t handle() const {
		return handle_;
	}
private:
	std::uint32_t handle_;
	int fd_;
};

template<typename It>
Buffer_list::Buffer_list(int fd, It b, It e) : fd_(fd)
{

	auto entries = std::make_unique<drm_amdgpu_bo_list_entry[]>(e - b);
	auto cur_entry = &entries[0];
	for(auto it = b; it != e; ++it, ++cur_entry) {
		cur_entry->bo_handle = (*it)->handle();
		cur_entry->bo_priority = 0;
	}
	drm_amdgpu_bo_list args = { };
	args.in.operation = AMDGPU_BO_LIST_OP_CREATE;
	args.in.bo_number = e - b;
	args.in.bo_info_size = sizeof(struct drm_amdgpu_bo_list_entry);
	args.in.bo_info_ptr = reinterpret_cast<std::uint64_t>(entries.get());

	if (drmCommandWriteRead(fd, DRM_AMDGPU_BO_LIST, &args, sizeof(args))) {
		throw -1;
	}
	handle_ = args.out.list_handle;
}

Buffer_list::~Buffer_list()
{
	union drm_amdgpu_bo_list free_args = { };

	free_args.in.operation = AMDGPU_BO_LIST_OP_DESTROY;
	free_args.in.list_handle = handle_;

	if (drmCommandWriteRead(fd_, DRM_AMDGPU_BO_LIST, &free_args,
			sizeof(free_args))) {
		std::abort();
	}
}
}

void Command_buffer::wait() {
	if(submit_handle_ == ~0ull)
		std::abort();

	union drm_amdgpu_wait_cs args = { };

	timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	args.in.handle = submit_handle_;
	args.in.ip_type = hw_type_;
	args.in.ip_instance = 0;
	args.in.ring = queue_;
	args.in.ctx_id = device_->ctx;
	args.in.timeout = ts.tv_sec * 1000000000ULL + ts.tv_nsec + 10000000000ull;
	if (drmIoctl(device_->fd, DRM_IOCTL_AMDGPU_WAIT_CS, &args)) {
		std::cerr << "cs wait failed\n";
		std::abort();
	}
	if(args.out.status)
		std::cout << "wait failed\n";
}

void Command_buffer::submit_internal(std::vector<std::uint32_t> const* const* ibs,
                                     unsigned const* flags, unsigned count)
{
	if(submit_handle_ != ~0ull)
		std::abort();

	std::uint64_t chunk_pointers[3] = {};
	drm_amdgpu_cs_chunk chunks[3] = {};
	drm_amdgpu_cs_chunk_data chunks_data[3] = {};
	drm_amdgpu_cs cs_args = { };
	cs_args.in.ctx_id = device_->ctx;
	cs_args.in.chunks = reinterpret_cast<std::uint64_t>(&chunk_pointers[0]);

	for(unsigned i = 0; i < count; ++i) {
		auto ib = std::make_shared<Buffer>(*device_, ibs[i]->size() * sizeof(std::uint32_t), Buffer::gtt);
		upload(*ib, ibs[i]->data(), 0, ibs[i]->size() * sizeof(std::uint32_t));
		buffers_.push_back(ib);

		chunks_data[i].ib_data.va_start = ib->gpu_address();
		chunks_data[i].ib_data.ib_bytes = ibs[i]->size() * sizeof(std::uint32_t);
		chunks_data[i].ib_data.ip_type = hw_type_;
		chunks_data[i].ib_data.ip_instance = 0;
		chunks_data[i].ib_data.ring = queue_;
		chunks_data[i].ib_data.flags = flags[i];

		chunk_pointers[i] = reinterpret_cast<std::uint64_t>(&chunks[i]);
		chunks[i].chunk_id = AMDGPU_CHUNK_ID_IB;
		chunks[i].length_dw = sizeof(drm_amdgpu_cs_chunk_ib) / 4;
		chunks[i].chunk_data = reinterpret_cast<std::uint64_t>(&chunks_data[i]);
	}

	Buffer_list buf_list{device_->fd, buffers_.begin(), buffers_.end()};
	cs_args.in.bo_list_handle = buf_list.handle();
	cs_args.in.num_chunks = count;
	if (drmCommandWriteRead(device_->fd, DRM_AMDGPU_CS, &cs_args, sizeof(cs_args))) {
		std::cout << "submit failed" << "\n";
		throw -1;
	}

	submit_handle_ = cs_args.out.handle;
}

void Command_buffer::add_buffer(std::shared_ptr<Buffer> buf) {
	if(!buf)
		return;
	for(auto& e : buffers_)
		if(e == buf)
			return;
	buffers_.push_back(std::move(buf));
}

Time_elapsed_query::Time_elapsed_query(Device& dev) {
	buffer = std::make_shared<Buffer>(dev, 16, Buffer::gtt);
	frequency = dev.gpu_counter_freq();
}
#define PKT3_RELEASE_MEM 0x49
#define WAIT_REG_MEM_MEM_SPACE(x) (((x)&0x3) << 4)

std::uint64_t Time_elapsed_query::value() {
	auto data = reinterpret_cast<std::uint64_t const*>(buffer->map());
	auto ret = data[1] - data[0];
	buffer->unmap();

	return ret * 100000 / frequency;
}

Perf_counter_query::Perf_counter_query(Device& dev, Perf_counter_id const* counters_data, unsigned count) : counters(counters_data, counters_data + count), offset{0} {
	buffer = std::make_shared<Buffer>(dev, 16, Buffer::gtt);
	std::sort(counters.begin(), counters.end(), [](auto a, auto b) { if(a.block != b.block) return a.block < b.block; return a.counter < b.counter; });
}

void Perf_counter_query::finalize() {
	auto ptr = reinterpret_cast<std::uint64_t const*>(buffer->map());
	auto ptr2 = ptr + read2_base / 8;
	for(auto e : counters) {
		switch(e.block) {
		case Perf_counter_id::GRBM:
		case Perf_counter_id::SQ:
			std::cout << ptr[0] << " " << ptr[1] << " " << ptr[2] << " " << ptr[3] << " : " << ptr2[0] << " " << ptr2[1] << " " << ptr2[2] << " " << ptr2[3] << "\n";
			ptr += 4;
			ptr2 += 4;
			break;
		case Perf_counter_id::TCC:
			std::cout << ptr[0] << " " << ptr[1] << " " << ptr[2] << " " << ptr[3] << " : " << ptr2[0] << " " << ptr2[1] << " " << ptr2[2] << " " << ptr2[3] << "\n";
			std::cout << "  " <<ptr[4] << " " << ptr[5] << " " << ptr[6] << " " << ptr[7] << " : " << ptr2[4] << " " << ptr2[5] << " " << ptr2[6] << " " << ptr2[7] << "\n";
			std::cout << "  " <<ptr[8] << " " << ptr[9] << " " << ptr[10] << " " << ptr[11] << " : " << ptr2[8] << " " << ptr2[9] << " " << ptr2[10] << " " << ptr2[11] << "\n";
			std::cout << "  " <<ptr[12] << " " << ptr[13] << " " << ptr[14] << " " << ptr[15] << " : " << ptr2[12] << " " << ptr2[13] << " " << ptr2[14] << " " << ptr2[15] << "\n";
			ptr += 16;
			ptr2 += 16;
			break;
		}
	}

	buffer->unmap();
}

Compute_shader::Compute_shader(Device& dev, Compute_shader_create_args const& args) {
	buffer = std::make_shared<Buffer>(dev, args.size * 4, Buffer::vram);
	upload(*buffer, args.data, 0, args.size * 4);
	scratch_size = args.scratch_size;
	rsrc1 = S_00B848_VGPRS((args.vgpr_count - 1) / 4) |
	        S_00B848_SGPRS((args.sgpr_count - 1) / 8) |
	        S_00B848_DX10_CLAMP(1) |
	        S_00B848_FLOAT_MODE(V_00B028_FP_64_DENORMS);
	rsrc2 = S_00B84C_USER_SGPR(args.user_sgpr_count) |
	        S_00B84C_SCRATCH_EN(!!args.scratch_size) |
	        S_00B84C_TGID_X_EN(args.tgid_x_en) | S_00B84C_TGID_Y_EN(args.tgid_y_en) |
	        S_00B84C_TGID_Z_EN(args.tgid_z_en) | S_00B84C_TIDIG_COMP_CNT(args.tidig_comp_cnt) |
	        S_00B84C_LDS_SIZE((args.lds_size + 511) / 512);

}
Compute_command_buffer::Compute_command_buffer(Device& dev, unsigned queue) : Command_buffer{dev, AMDGPU_HW_IP_COMPUTE, queue}
{

	start_emit_sh_reg(*this, R_00B810_COMPUTE_START_X, 3);
	emit(0);
	emit(0);
	emit(0);

	start_emit_sh_reg(*this, R_00B854_COMPUTE_RESOURCE_LIMITS, 3);
	emit(0);
	emit(S_00B858_SH0_CU_EN(0xffff) | S_00B858_SH1_CU_EN(0xffff));
	emit(S_00B85C_SH0_CU_EN(0xffff) | S_00B85C_SH1_CU_EN(0xffff));

	start_emit_sh_reg(*this, R_00B864_COMPUTE_STATIC_THREAD_MGMT_SE2, 2);
	emit(S_00B864_SH0_CU_EN(0xffff) | S_00B864_SH1_CU_EN(0xffff));
	emit(S_00B868_SH0_CU_EN(0xffff) | S_00B868_SH1_CU_EN(0xffff));
}

void Compute_command_buffer::submit()
{
	emit(PKT3(PKT3_EVENT_WRITE, 0, 0));
	emit(EVENT_TYPE(V_028A90_CS_PARTIAL_FLUSH) | EVENT_INDEX(4));


	std::vector<std::uint32_t> const* ibs[1] = {};
	unsigned flags[1] = {};
	unsigned count = 0;

	while(data_.empty() || (data_.size() & 7))
		data_.push_back(0xffff1000u);

	ibs[count] = &data_;
	flags[count] = 0;
	++count;

	submit_internal(ibs, flags, count);
}

void Compute_command_buffer::set_compute_shader(Compute_shader&  shader)
{
	add_buffer(shader.buffer);

	uint64_t shader_va = shader.buffer->gpu_address();
	start_emit_sh_reg(*this, R_00B830_COMPUTE_PGM_LO, 2);
	emit(shader_va >> 8);
	emit(shader_va >> 40);

	start_emit_sh_reg(*this, R_00B848_COMPUTE_PGM_RSRC1, 2);
	emit(shader.rsrc1);
	emit(shader.rsrc2);

	start_emit_sh_reg(*this, R_00B860_COMPUTE_TMPRING_SIZE, 1);
	emit(S_00B860_WAVES(64) | S_00B860_WAVESIZE((shader.scratch_size + 1023) >> 10));
}

void Compute_command_buffer::set_compute_user_sgprs(unsigned start, unsigned count, std::uint32_t const* data) {
	start_emit_sh_reg(*this, R_00B900_COMPUTE_USER_DATA_0 + start * 4, count);
	for(unsigned i = 0; i < count; ++i)
		emit(data[i]);
}
void Compute_command_buffer::dispatch(std::array<std::size_t, 3> local_size, std::array<std::size_t, 3> global_size)
{
	start_emit_sh_reg(*this, R_00B81C_COMPUTE_NUM_THREAD_X, 3);
	for(int i = 0; i < 3; ++i)
		emit(S_00B81C_NUM_THREAD_FULL(local_size[i]) | S_00B81C_NUM_THREAD_PARTIAL((global_size[i] - 1) % local_size[i] + 1));

	emit(PKT3(PKT3_DISPATCH_DIRECT, 3, false) | PKT3_SHADER_TYPE_S(1));
	for(int i = 0 ; i < 3; ++i)
		emit((global_size[i] + local_size[i] - 1) / local_size[i]);
	emit(S_00B800_COMPUTE_SHADER_EN(1) | S_00B800_PARTIAL_TG_EN(1));
}


void Compute_command_buffer::begin(Time_elapsed_query& query) {
	emit(PKT3(PKT3_EVENT_WRITE, 0, 0));
	emit(EVENT_TYPE(V_028A90_CS_PARTIAL_FLUSH) | EVENT_INDEX(4));

	auto va = query.buffer->gpu_address();

	emit(PKT3(PKT3_RELEASE_MEM, 5, 0) | PKT3_SHADER_TYPE_S(1));
	emit(EVENT_TYPE(EVENT_TYPE_CACHE_FLUSH_AND_INV_TS_EVENT) | EVENT_INDEX(5));
	emit((3 << 29));
	emit(va);
	emit(va >> 32UL);
	emit(0);
	emit(0);
	add_buffer(query.buffer);
}
void Compute_command_buffer::end(Time_elapsed_query& query) {
	emit(PKT3(PKT3_EVENT_WRITE, 0, 0));
	emit(EVENT_TYPE(V_028A90_CS_PARTIAL_FLUSH) | EVENT_INDEX(4));

	auto va = query.buffer->gpu_address() + 8;

	emit(PKT3(PKT3_RELEASE_MEM, 5, 0) | PKT3_SHADER_TYPE_S(1));
	emit(EVENT_TYPE(EVENT_TYPE_CACHE_FLUSH_AND_INV_TS_EVENT) | EVENT_INDEX(5));
	emit((3 << 29));
	emit(va);
	emit(va >> 32UL);
	emit(0);
	emit(0);
}

#define EVENT_TYPE_SAMPLE_PIPELINESTAT 30
#define EVENT_TYPE_PERFCOUNTER_START 0x17
#define EVENT_TYPE_PERFCOUNTER_STOP 0x18
#define EVENT_TYPE_PERFCOUNTER_SAMPLE 0x1B


template<typename T>
void read(T& cmd_buf, Perf_counter_query& query) {
	unsigned sq_count = 0;
	unsigned grbm_count = 0;
	unsigned tcc_count = 0;
	cmd_buf.add_buffer(query.buffer);
	for(auto e : query.counters) {
		switch(e.block) {
			case Perf_counter_id::SQ:
			for(unsigned i = 0; i < 4; ++i) {
				start_emit_uconfig_reg(cmd_buf, R_030800_GRBM_GFX_INDEX, 1);
				cmd_buf.emit(S_030800_SH_BROADCAST_WRITES(1) | S_030800_SE_INDEX(i) |
				     S_030800_INSTANCE_BROADCAST_WRITES(1));

				std::uint64_t va = query.buffer->gpu_address() + query.offset + i * 8;
				cmd_buf.emit(PKT3(PKT3_COPY_DATA, 4, 0) | PKT3_SHADER_TYPE_S(1));
				cmd_buf.emit(COPY_DATA_SRC_SEL(COPY_DATA_PERF) | COPY_DATA_DST_SEL(5));
				cmd_buf.emit((R_034700_SQ_PERFCOUNTER0_LO + 8 * sq_count) >> 2);
				cmd_buf.emit(0); /* unused */
				cmd_buf.emit(va);
				cmd_buf.emit(va >> 32);

				cmd_buf.emit(PKT3(PKT3_COPY_DATA, 4, 0) | PKT3_SHADER_TYPE_S(1));
				cmd_buf.emit(COPY_DATA_SRC_SEL(COPY_DATA_PERF) | COPY_DATA_DST_SEL(5));
				cmd_buf.emit((R_034700_SQ_PERFCOUNTER0_LO + 8 * sq_count + 4) >> 2);
				cmd_buf.emit(0); /* unused */
				cmd_buf.emit((va + 4));
				cmd_buf.emit((va + 4) >> 32);
			}
			query.offset += 32;
			sq_count++;
			break;
			case Perf_counter_id::GRBM:
			for(unsigned i = 0; i < 4; ++i) {
				start_emit_uconfig_reg(cmd_buf, R_030800_GRBM_GFX_INDEX, 1);
				cmd_buf.emit(S_030800_SH_BROADCAST_WRITES(1) | S_030800_SE_INDEX(i) |
				     S_030800_INSTANCE_BROADCAST_WRITES(1));

				std::uint64_t va = query.buffer->gpu_address() + query.offset + i * 8;
				cmd_buf.emit(PKT3(PKT3_COPY_DATA, 4, 0) | PKT3_SHADER_TYPE_S(1));
				cmd_buf.emit(COPY_DATA_SRC_SEL(COPY_DATA_PERF) | COPY_DATA_DST_SEL(5));
				cmd_buf.emit((R_034100_GRBM_PERFCOUNTER0_LO + 12 * grbm_count) >> 2);
				cmd_buf.emit(0); /* unused */
				cmd_buf.emit(va);
				cmd_buf.emit(va >> 32);

				cmd_buf.emit(PKT3(PKT3_COPY_DATA, 4, 0) | PKT3_SHADER_TYPE_S(1));
				cmd_buf.emit(COPY_DATA_SRC_SEL(COPY_DATA_PERF) | COPY_DATA_DST_SEL(5));
				cmd_buf.emit((R_034100_GRBM_PERFCOUNTER0_LO + 12 * grbm_count + 4) >> 2);
				cmd_buf.emit(0); /* unused */
				cmd_buf.emit((va + 4));
				cmd_buf.emit((va + 4) >> 32);
			}
			query.offset += 32;
			grbm_count++;
			break;
			case Perf_counter_id::TCC:
			for(unsigned i = 0; i < 16; ++i) {
				start_emit_uconfig_reg(cmd_buf, R_030800_GRBM_GFX_INDEX, 1);
				cmd_buf.emit(S_030800_SH_BROADCAST_WRITES(1) | S_030800_SE_BROADCAST_WRITES(1) |
				     S_030800_INSTANCE_INDEX(i));

				std::uint64_t va = query.buffer->gpu_address() + query.offset + i * 8;
				cmd_buf.emit(PKT3(PKT3_COPY_DATA, 4, 0) | PKT3_SHADER_TYPE_S(1));
				cmd_buf.emit(COPY_DATA_SRC_SEL(COPY_DATA_PERF) | COPY_DATA_DST_SEL(5));
				cmd_buf.emit((R_034E00_TCC_PERFCOUNTER0_LO + 8 * tcc_count) >> 2);
				cmd_buf.emit(0); /* unused */
				cmd_buf.emit(va);
				cmd_buf.emit(va >> 32);

				cmd_buf.emit(PKT3(PKT3_COPY_DATA, 4, 0) | PKT3_SHADER_TYPE_S(1));
				cmd_buf.emit(COPY_DATA_SRC_SEL(COPY_DATA_PERF) | COPY_DATA_DST_SEL(5));
				cmd_buf.emit((R_034E00_TCC_PERFCOUNTER0_LO + 8 * tcc_count + 4) >> 2);
				cmd_buf.emit(0); /* unused */
				cmd_buf.emit((va + 4));
				cmd_buf.emit((va + 4) >> 32);
			}
			query.offset += 128;
			tcc_count++;
			break;
		}
	}
	start_emit_uconfig_reg(cmd_buf, R_030800_GRBM_GFX_INDEX, 1);
	cmd_buf.emit(S_030800_SH_BROADCAST_WRITES(1) | S_030800_SE_BROADCAST_WRITES(1) |
             S_030800_INSTANCE_BROADCAST_WRITES(1));
}
void Compute_command_buffer::begin(Perf_counter_query& query) {

	read(*this, query);
	start_emit_uconfig_reg(*this, R_036780_SQ_PERFCOUNTER_CTRL, 2);
	emit(0x7F);
	emit(0xFFFFFFFFU);

	start_emit_uconfig_reg(*this,R_030800_GRBM_GFX_INDEX, 1);
	emit(S_030800_SH_BROADCAST_WRITES(1) | S_030800_SE_BROADCAST_WRITES(1) |
              S_030800_INSTANCE_BROADCAST_WRITES(1));

	emit(PKT3(PKT3_EVENT_WRITE, 0, 0));
	emit(EVENT_TYPE(V_028A90_CS_PARTIAL_FLUSH) | EVENT_INDEX(4));

	start_emit_uconfig_reg(*this,R_036020_CP_PERFMON_CNTL, 1);
	emit(S_036020_PERFMON_STATE(V_036020_DISABLE_AND_RESET));

	emit(PKT3(PKT3_EVENT_WRITE, 0, 0));
	emit(EVENT_TYPE(EVENT_TYPE_PERFCOUNTER_START) | EVENT_INDEX(0));

	start_emit_uconfig_reg(*this,R_036020_CP_PERFMON_CNTL, 1);
	emit(S_036020_PERFMON_STATE(V_036020_START_COUNTING) | S_036020_PERFMON_SAMPLE_ENABLE(1));

	for(auto it = query.counters.begin(); it != query.counters.end();) {
		auto it2 = it;
		while(it2 != query.counters.end() && it2->block == it->block)
			++it2;
		switch(it->block) {
			case Perf_counter_id::SQ:
				start_emit_uconfig_reg(*this, R_036700_SQ_PERFCOUNTER0_SELECT, it2 - it);
				for(unsigned i = 0; i < (it2 - it); ++i)
					emit(S_036700_SQC_BANK_MASK(15) | S_036700_SQC_CLIENT_MASK(15) |
					     S_036700_SIMD_MASK(15) | S_036700_PERF_SEL(it[i].counter));
				break;
			case Perf_counter_id::GRBM:
				start_emit_uconfig_reg(*this, R_036100_GRBM_PERFCOUNTER0_SELECT, it2 - it);
				for(unsigned i = 0; i < (it2 - it); ++i) {
					emit(S_036100_PERF_SEL(it[i].counter));
				}
				break;
			case Perf_counter_id::TCC:
				start_emit_uconfig_reg(*this, R_036E00_TCC_PERFCOUNTER0_SELECT, it2 - it + std::min<unsigned>(2, it2 - it));
				for(unsigned i = 0; i < (it2 - it); ++i) {
					emit(S_036E00_PERF_SEL(it[i].counter));
					if(i < 2) emit(0);
				}
				break;
		}

		it = it2;
	}
	read(*this, query);
	query.read2_base = query.offset;
}

void Compute_command_buffer::end(Perf_counter_query& query) {
	emit(PKT3(PKT3_EVENT_WRITE, 0, 0));
	emit(EVENT_TYPE(EVENT_TYPE_PERFCOUNTER_SAMPLE) | EVENT_INDEX(0));

	emit(PKT3(PKT3_EVENT_WRITE, 0, 0));
	emit(EVENT_TYPE(EVENT_TYPE_PERFCOUNTER_STOP) | EVENT_INDEX(0));

	start_emit_uconfig_reg(*this,R_036020_CP_PERFMON_CNTL, 1);
	emit(S_036020_PERFMON_STATE(V_036020_STOP_COUNTING) | S_036020_PERFMON_SAMPLE_ENABLE(1));

	read(*this, query);
}


Graphics_command_buffer::Graphics_command_buffer(Device& dev, unsigned queue) : Command_buffer{dev, AMDGPU_HW_IP_GFX, queue}
{
}

void Graphics_command_buffer::submit()
{
	std::vector<std::uint32_t> const* ibs[3] = {};
	unsigned flags[3] = {};
	unsigned count = 0;

	if(!ce_preamble_data_.empty()) {

		while(ce_preamble_data_.size() & 7)
			ce_preamble_data_.push_back(0xffff1000u);

		ibs[count] = &ce_preamble_data_;
		flags[count] = AMDGPU_IB_FLAG_CE | AMDGPU_IB_FLAG_PREAMBLE;
		++count;
	}
	if(!ce_data_.empty()) {

		while(ce_data_.size() & 7)
			ce_data_.push_back(0xffff1000u);

		ibs[count] = &ce_data_;
		flags[count] = AMDGPU_IB_FLAG_CE;
		++count;
	}

	while(data_.empty() || (data_.size() & 7))
		data_.push_back(0xffff1000u);

	ibs[count] = &data_;
	flags[count] = 0;
	++count;

	submit_internal(ibs, flags, count);
}

void* Graphics_command_buffer::ce_write(std::size_t offset, std::size_t size) {
	auto words = size / 4;
	ce_emit(PKT3(PKT3_WRITE_CONST_RAM, words, 0));
	ce_emit(offset);
	auto index = ce_data_.size();
	ce_data_.resize(index + words);
	return &ce_data_[index];
}

void Graphics_command_buffer::ce_dump(std::size_t ce_offset, std::shared_ptr<Buffer> buf, std::size_t offset, std::size_t size) {
	auto va = buf->gpu_address() + offset;
	ce_emit(PKT3(PKT3_DUMP_CONST_RAM, 3, 0));
	ce_emit(ce_offset);
	ce_emit(size / 4);
	ce_emit(va);
	ce_emit(va >> 32);
}

void Graphics_command_buffer::ce_pre_draw_sync()
{
	ce_emit(PKT3(PKT3_INCREMENT_CE_COUNTER, 0, 0));
	ce_emit(1);

	emit(PKT3(PKT3_WAIT_ON_CE_COUNTER, 0, 0));
	emit(1);
}

void Graphics_command_buffer::ce_post_draw_sync()
{
	emit(PKT3(PKT3_INCREMENT_DE_COUNTER, 0, 0));
	emit(0);
}


namespace {
std::size_t get_hw_instance_count(int fd, unsigned hw_type)
{
	drm_amdgpu_info request = {};
	std::uint64_t ret = 0;

	request.query = AMDGPU_INFO_HW_IP_COUNT;
	request.return_pointer = reinterpret_cast<std::uintptr_t>(&ret);
	request.return_size = sizeof(ret);
	request.query_hw_ip.type = hw_type;

	if (drmCommandWrite(fd, DRM_AMDGPU_INFO, &request,
	                    sizeof(struct drm_amdgpu_info))) {
		throw -1;
	}
	return ret;
}

void get_hw_queue_info(int fd, unsigned hw_type, unsigned instance, std::vector<std::pair<unsigned, unsigned>>& queues)
{
	drm_amdgpu_info request = {};
	drm_amdgpu_info_hw_ip ret = {};

	request.query = AMDGPU_INFO_HW_IP_INFO;
	request.return_pointer = reinterpret_cast<std::uintptr_t>(&ret);
	request.return_size = sizeof(ret);
	request.query_hw_ip.type = hw_type;
	request.query_hw_ip.ip_instance = instance;

	if (drmCommandWrite(fd, DRM_AMDGPU_INFO, &request,
	                    sizeof(struct drm_amdgpu_info))) {
		throw -1;
	}

	for(unsigned i = 0; i < 32; ++i) {
		if(ret.available_rings & (1u << i))
			queues.push_back({instance, i});
	}
}
}
Device::Device()
{
	fd = open("/dev/dri/renderD128", O_RDWR);
	if(fd < 0) {
		throw -1;
	}
	{
		struct drm_amdgpu_info request = { };

		request.return_pointer = reinterpret_cast<std::uintptr_t>(&device_info_);
		request.return_size = sizeof(drm_amdgpu_info_device);
		request.query = AMDGPU_INFO_DEV_INFO;

		if (drmCommandWrite(fd, DRM_AMDGPU_INFO, &request,
				sizeof(struct drm_amdgpu_info))) {
			close(fd);
			throw -1;
		}

	}

	{
		drm_amdgpu_ctx alloc_args;
		alloc_args.in.op = AMDGPU_CTX_OP_ALLOC_CTX;
		if (drmCommandWriteRead(fd, DRM_AMDGPU_CTX, &alloc_args,
			sizeof(alloc_args))) {
			close(fd);
			throw -1;
		}

		ctx = alloc_args.out.alloc.ctx_id;
	}
	auto gfx_instances = get_hw_instance_count(fd, AMDGPU_HW_IP_GFX);
	auto compute_instances = get_hw_instance_count(fd, AMDGPU_HW_IP_COMPUTE);
	auto dma_instances = get_hw_instance_count(fd, AMDGPU_HW_IP_DMA);
	for(unsigned i = 0; i < gfx_instances; ++i)
		get_hw_queue_info(fd, AMDGPU_HW_IP_GFX, i, gfx_rings_);
	for(unsigned i = 0; i < compute_instances; ++i)
		get_hw_queue_info(fd, AMDGPU_HW_IP_COMPUTE, i, compute_rings_);
	for(unsigned i = 0; i < dma_instances; ++i)
		get_hw_queue_info(fd, AMDGPU_HW_IP_DMA, i, dma_rings_);
}

Device::~Device()
{
	{
		drm_amdgpu_ctx free_args;
		free_args.in.op = AMDGPU_CTX_OP_FREE_CTX;
		free_args.in.ctx_id = ctx;
		drmCommandWriteRead(fd, DRM_AMDGPU_CTX, &free_args, sizeof(free_args));
	}
	close(fd);
}
