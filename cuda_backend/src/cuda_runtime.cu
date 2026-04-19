#include "dtf/cuda_backend.hpp"

#include <cuda_runtime_api.h>

#include <sstream>
#include <stdexcept>

namespace dtf {

namespace {

std::runtime_error cuda_error(const char* message, cudaError_t status) {
  std::ostringstream oss;
  oss << message << ": " << cudaGetErrorString(status);
  return std::runtime_error(oss.str());
}

}  // namespace

std::vector<DeviceInfo> CudaBackend::enumerate_devices() const {
  int device_count = 0;
  const cudaError_t count_status = cudaGetDeviceCount(&device_count);
  if (count_status != cudaSuccess) {
    throw cuda_error("cudaGetDeviceCount failed", count_status);
  }

  std::vector<DeviceInfo> devices;
  devices.reserve(static_cast<std::size_t>(device_count));
  for (int device_index = 0; device_index < device_count; ++device_index) {
    cudaDeviceProp props{};
    const cudaError_t props_status = cudaGetDeviceProperties(&props, device_index);
    if (props_status != cudaSuccess) {
      throw cuda_error("cudaGetDeviceProperties failed", props_status);
    }
    devices.push_back(DeviceInfo{
        .device_index = device_index,
        .name = props.name,
        .major_cc = props.major,
        .minor_cc = props.minor,
        .total_global_mem_bytes = props.totalGlobalMem,
        .multiprocessor_count = props.multiProcessorCount,
    });
  }
  return devices;
}

EngineTelemetry CudaBackend::configure_runtime(const RuntimeOptions& options) {
  if (options.memory_pool_release_threshold_bytes == 0) {
    throw std::invalid_argument("memory_pool_release_threshold_bytes must be > 0");
  }
  int least_priority = 0;
  int greatest_priority = 0;
  cudaError_t status = cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);
  if (status != cudaSuccess) {
    throw cuda_error("cudaDeviceGetStreamPriorityRange failed", status);
  }
  if (options.compute_stream_priority < greatest_priority || options.compute_stream_priority > least_priority) {
    throw std::invalid_argument("compute_stream_priority is outside the CUDA-supported priority range");
  }
  if (options.memcpy_stream_priority < greatest_priority || options.memcpy_stream_priority > least_priority) {
    throw std::invalid_argument("memcpy_stream_priority is outside the CUDA-supported priority range");
  }

  cudaStream_t compute_stream = nullptr;
  cudaStream_t memcpy_stream = nullptr;

  status = cudaStreamCreateWithPriority(&compute_stream, cudaStreamNonBlocking, options.compute_stream_priority);
  if (status != cudaSuccess) {
    throw cuda_error("cudaStreamCreateWithPriority(compute) failed", status);
  }

  status = cudaStreamCreateWithPriority(&memcpy_stream, cudaStreamNonBlocking, options.memcpy_stream_priority);
  if (status != cudaSuccess) {
    cudaStreamDestroy(compute_stream);
    throw cuda_error("cudaStreamCreateWithPriority(memcpy) failed", status);
  }

  cudaMemPool_t mempool = nullptr;
  bool async_allocator_available = false;
  bool memory_pool_enabled = false;
  if (options.enable_async_pool) {
    status = cudaDeviceGetDefaultMemPool(&mempool, 0);
    if (status == cudaSuccess) {
      async_allocator_available = true;
      status = cudaMemPoolSetAttribute(
          mempool,
          cudaMemPoolAttrReleaseThreshold,
          &options.memory_pool_release_threshold_bytes);
      if (status == cudaSuccess) {
        memory_pool_enabled = true;
      }
    }
  }

  if (compute_stream_ != nullptr) {
    cudaStreamDestroy(static_cast<cudaStream_t>(compute_stream_));
  }
  if (memcpy_stream_ != nullptr) {
    cudaStreamDestroy(static_cast<cudaStream_t>(memcpy_stream_));
  }

  compute_stream_ = compute_stream;
  memcpy_stream_ = memcpy_stream;
  mempool_ = mempool;

  return EngineTelemetry{
      .async_allocator_available = async_allocator_available,
      .memory_pool_enabled = memory_pool_enabled,
      .separate_streams_configured = true,
      .profiling_required_before_optimization = true,
      .runtime_options_accepted = true,
      .notes = "Fuse small kernels, prefer cuBLASLt and CUTLASS GEMM paths, and profile with Nsight Compute before micro-optimizing.",
  };
}

}  // namespace dtf
