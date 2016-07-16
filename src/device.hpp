#ifndef DEVICE_HPP
#define DEVICE_HPP

#include <cassert>
#include <cstdint>
#include <memory>
#include <vector>

#include <amdgpu_drm.h>

#include "sid.h"

class Device;
class Buffer;

class Buffer {
public:
	enum {vram = 1, gtt = 2};

	Buffer(Device& device, std::uint64_t size, unsigned options);
	~Buffer();

	void* map();
	void unmap();

	std::uint32_t handle() { return handle_; }
	std::uint64_t gpu_address() { return gpu_address_; }
private:
	static std::uint64_t alloc_va(Device& dev, std::uint64_t size);
	static void free_va(Device& dev, std::uint64_t address, std::uint64_t size);

	Device* device_;
	std::uint64_t size_, gpu_address_;
	std::uint32_t handle_;
	unsigned map_count_;
	void* map_pointer_;
};

void upload(Buffer& buf, void const* data, std::uint64_t offset, std::size_t size);

class Command_buffer {
public:
	Command_buffer(Device& dev, unsigned hw_type, unsigned queue);
	virtual ~Command_buffer() {}

	virtual void submit() = 0;
	void wait();

	unsigned hw_type() const { return hw_type_; }
	Device& device() { return *device_; }



	void add_buffer(std::shared_ptr<Buffer> buf);

protected:
	void submit_internal(std::vector<std::uint32_t> const* const* ibs,
	                     unsigned const* flags, unsigned count);

private:
	Device* device_;
	unsigned hw_type_;
	unsigned queue_;
	std::uint64_t submit_handle_;
	std::vector<std::shared_ptr<Buffer>> buffers_;



};

struct Time_elapsed_query {
	Time_elapsed_query(Device& dev);

	std::uint64_t value();

	std::shared_ptr<Buffer> buffer;
	std::uint64_t frequency;
};

struct Perf_counter_id {
	enum {
		GRBM,
		SQ,
		TCC
	} block;
	unsigned counter;
};

struct Perf_counter_query {
	Perf_counter_query(Device& dev, Perf_counter_id const* counters, unsigned count);

	void finalize();
	std::shared_ptr<Buffer> buffer;
	std::vector<Perf_counter_id> counters;
	unsigned offset;
	unsigned read2_base;
};


struct Compute_shader_create_args {
	const std::uint32_t* data;
	unsigned size;

	unsigned vgpr_count;
	unsigned sgpr_count;
	unsigned lds_size;

	unsigned user_sgpr_count;
	bool tgid_x_en, tgid_y_en, tgid_z_en;
	bool tg_size_en;

	unsigned tidig_comp_cnt;

	unsigned scratch_size;
};
struct Compute_shader {
	Compute_shader(Device& dev, Compute_shader_create_args const& args);

	std::shared_ptr<Buffer> buffer;
	unsigned rsrc1;
	unsigned rsrc2;
	unsigned scratch_size;

};

class Compute_command_buffer final : public Command_buffer {
public:
	Compute_command_buffer(Device& dev, unsigned queue);

	void submit() override;

	void set_compute_shader(Compute_shader&  shader);

	void set_compute_user_sgprs(unsigned start, unsigned count, std::uint32_t const* data);
	void dispatch(std::array<std::size_t, 3> local_size, std::array<std::size_t, 3> global_size);

	void begin(Time_elapsed_query& query);
	void end(Time_elapsed_query& query);
	void begin(Perf_counter_query& query);
	void end(Perf_counter_query& query);

	void emit(std::uint32_t v) { data_.push_back(v); }
private:
	std::vector<std::uint32_t> data_;
};

class Graphics_command_buffer final : public Command_buffer {
public:
	Graphics_command_buffer(Device& dev, unsigned queue);

	void submit() override;

	void* ce_write(std::size_t offset, std::size_t size);
	void ce_dump(std::size_t ce_offset, std::shared_ptr<Buffer> buf, std::size_t offset, std::size_t size);
	void ce_pre_draw_sync();
	void ce_post_draw_sync();

	void emit(std::uint32_t v) { data_.push_back(v); }

	void ce_emit(std::uint32_t v) { ce_data_.push_back(v); }
	void preamble_emit(std::uint32_t v) { ce_preamble_data_.push_back(v); }
private:
	std::vector<std::uint32_t> ce_data_, ce_preamble_data_, data_;
};

template<typename T>
void start_emit_sh_reg(T& cmd_buf, unsigned reg, unsigned count = 1) {
	assert(reg >= SI_SH_REG_OFFSET && reg + 4 * count < SI_SH_REG_END);
	cmd_buf.emit(PKT3(PKT3_SET_SH_REG, count, 0));
	cmd_buf.emit((reg - SI_SH_REG_OFFSET ) >> 2);
}

template<typename T>
void start_emit_context_reg(T& cmd_buf, unsigned reg, unsigned count = 1) {
	assert(reg >= SI_CONTEXT_REG_OFFSET && reg + 4 * count < SI_CONTEXT_REG_END);
	cmd_buf.emit(PKT3(PKT3_SET_CONTEXT_REG, count, 0));
	cmd_buf.emit((reg - SI_CONTEXT_REG_OFFSET ) >> 2);
}

template<typename T>
void start_emit_config_reg(T& cmd_buf, unsigned reg, unsigned count = 1) {
	assert(reg >= SI_CONFIG_REG_OFFSET && reg + 4 * count < SI_CONFIG_REG_END);
	cmd_buf.emit(PKT3(PKT3_SET_CONFIG_REG, count, 0));
	cmd_buf.emit((reg - SI_CONFIG_REG_OFFSET ) >> 2);
}

template<typename T>
void start_emit_uconfig_reg(T& cmd_buf, unsigned reg, unsigned count = 1) {
	assert(reg >= CIK_UCONFIG_REG_OFFSET && reg + 4 * count < CIK_UCONFIG_REG_END);
	cmd_buf.emit(PKT3(PKT3_SET_UCONFIG_REG, count, 0));
	cmd_buf.emit((reg - CIK_UCONFIG_REG_OFFSET ) >> 2);
}

class Device {
public:
	Device();
	~Device();

	unsigned graphics_queue_count() const { return gfx_rings_.size(); }
	unsigned compute_queue_count() const { return compute_rings_.size(); }
	unsigned dma_queue_count() const { return dma_rings_.size(); }
	std::uint64_t gpu_counter_freq() const { return device_info_.gpu_counter_freq; }
private:
	int fd;
	unsigned ctx;
	drm_amdgpu_info_device device_info_;
	std::vector<std::pair<std::uint64_t, std::uint64_t>> va_ranges;
	std::vector<std::pair<unsigned, unsigned>> gfx_rings_, compute_rings_, dma_rings_;
	friend class Buffer;
	friend class Command_buffer;
};

#endif
