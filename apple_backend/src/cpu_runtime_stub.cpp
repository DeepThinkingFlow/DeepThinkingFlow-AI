#include "dtf/apple_backend.hpp"

namespace dtf {

std::vector<AppleDeviceInfo> AppleBackend::enumerate_devices() const {
  return {};
}

AppleTelemetry AppleBackend::configure_runtime(const AppleRuntimeOptions&) {
  return AppleTelemetry{
      .binary_archive_expected = false,
      .unified_memory_assumed = false,
      .command_queue_required = false,
      .mps_graph_preferred = false,
      .runtime_options_accepted = false,
      .notes = "Apple Silicon backend is disabled in this build. This CPU fallback exists to unblock wrapper and API development on non-macOS hosts.",
  };
}

}  // namespace dtf
