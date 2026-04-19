#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace dtf {

struct BuildConfig {
  std::string cuda_architectures;
  bool cuda_enabled;
  bool flash_attention_enabled;
  bool bitsandbytes_enabled;
  bool nccl_enabled;
};

struct DeviceInfo {
  int device_index;
  std::string name;
  int major_cc;
  int minor_cc;
  std::size_t total_global_mem_bytes;
  int multiprocessor_count;
};

struct RuntimeOptions {
  int compute_stream_priority;
  int memcpy_stream_priority;
  bool enable_async_pool;
  std::size_t memory_pool_release_threshold_bytes;
};

struct EngineTelemetry {
  bool async_allocator_available;
  bool memory_pool_enabled;
  bool separate_streams_configured;
  bool profiling_required_before_optimization;
  bool runtime_options_accepted;
  std::string notes;
};

class CudaBackend {
 public:
  CudaBackend();
  ~CudaBackend();

  CudaBackend(const CudaBackend&) = delete;
  CudaBackend& operator=(const CudaBackend&) = delete;

  BuildConfig build_config() const;
  std::vector<DeviceInfo> enumerate_devices() const;
  RuntimeOptions default_runtime_options() const;
  EngineTelemetry configure_runtime(const RuntimeOptions& options);

 private:
  void* compute_stream_;
  void* memcpy_stream_;
  void* mempool_;
};

}  // namespace dtf
