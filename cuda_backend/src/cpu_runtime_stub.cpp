#include "dtf/cuda_backend.hpp"

#include <stdexcept>

namespace dtf {

std::vector<DeviceInfo> CudaBackend::enumerate_devices() const {
  return {};
}

EngineTelemetry CudaBackend::configure_runtime(const RuntimeOptions&) {
  return EngineTelemetry{
      .async_allocator_available = false,
      .memory_pool_enabled = false,
      .separate_streams_configured = false,
      .profiling_required_before_optimization = false,
      .runtime_options_accepted = false,
      .notes = "CUDA is disabled in this build. This CPU fallback exists to unblock wrapper/API development until an NVIDIA host is available.",
  };
}

}  // namespace dtf
