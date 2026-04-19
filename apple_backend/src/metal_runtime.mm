#include "dtf/apple_backend.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <CoreML/CoreML.h>

#include <stdexcept>

namespace dtf {

std::vector<AppleDeviceInfo> AppleBackend::enumerate_devices() const {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (device == nil) {
    return {};
  }

  AppleDeviceInfo info{
      .chip_family = std::string([[device name] UTF8String]),
      .unified_memory_bytes = static_cast<std::size_t>([[NSProcessInfo processInfo] physicalMemory]),
      .metal_available = true,
      .coreml_available = true,
  };
  return {info};
}

AppleTelemetry AppleBackend::configure_runtime(const AppleRuntimeOptions& options) {
  if (!options.use_mps_graph) {
    throw std::invalid_argument("Apple Silicon backend requires MPSGraph-first execution in strict mode.");
  }
  if (!options.use_binary_archive_cache) {
    throw std::invalid_argument("Apple Silicon backend requires binary archive caching to avoid first-load stutter.");
  }
  if (!options.prefer_unified_memory_views) {
    throw std::invalid_argument("Apple Silicon backend expects unified memory-aware tensor views.");
  }

  return AppleTelemetry{
      .binary_archive_expected = true,
      .unified_memory_assumed = true,
      .command_queue_required = true,
      .mps_graph_preferred = true,
      .runtime_options_accepted = true,
      .notes = "Prefer MLX native path, use MPSGraph over isolated MPS ops, avoid unnecessary tensor copies, and reserve Core ML conversion for ANE-targeted deployment.",
  };
}

}  // namespace dtf
