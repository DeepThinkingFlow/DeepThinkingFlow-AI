#include "dtf/cuda_backend.hpp"

#if DTF_CUDA_ENABLED
#include <cuda_runtime_api.h>
#endif

#include <cstdlib>
#include <stdexcept>

namespace dtf {

namespace {

std::string configured_architectures() {
#ifdef CMAKE_CUDA_ARCHITECTURES
  return CMAKE_CUDA_ARCHITECTURES;
#else
  const char* arch = std::getenv("CMAKE_CUDA_ARCHITECTURES");
  return arch == nullptr ? "" : std::string(arch);
#endif
}

}  // namespace

CudaBackend::CudaBackend() : compute_stream_(nullptr), memcpy_stream_(nullptr), mempool_(nullptr) {}

CudaBackend::~CudaBackend() {
#if DTF_CUDA_ENABLED
  if (compute_stream_ != nullptr) {
    cudaStreamDestroy(static_cast<cudaStream_t>(compute_stream_));
    compute_stream_ = nullptr;
  }
  if (memcpy_stream_ != nullptr) {
    cudaStreamDestroy(static_cast<cudaStream_t>(memcpy_stream_));
    memcpy_stream_ = nullptr;
  }
#endif
  mempool_ = nullptr;
}

BuildConfig CudaBackend::build_config() const {
  return BuildConfig{
      .cuda_architectures = configured_architectures(),
      .cuda_enabled =
#if DTF_CUDA_ENABLED
          true,
#else
          false,
#endif
#if DTF_CUDA_ENABLED
      .flash_attention_enabled =
#if DTF_ENABLE_FLASH_ATTENTION
          true,
#else
          false,
#endif
      .bitsandbytes_enabled =
#if DTF_ENABLE_BITSANDBYTES
          true,
#else
          false,
#endif
      .nccl_enabled =
#if DTF_ENABLE_NCCL
          true,
#else
          false,
#endif
#else
      .flash_attention_enabled =
          false,
      .bitsandbytes_enabled =
          false,
      .nccl_enabled =
          false,
#endif
  };
}

RuntimeOptions CudaBackend::default_runtime_options() const {
  return RuntimeOptions{
      .compute_stream_priority = 0,
      .memcpy_stream_priority = 0,
      .enable_async_pool = true,
      .memory_pool_release_threshold_bytes = 512ULL * 1024ULL * 1024ULL,
  };
}

}  // namespace dtf
