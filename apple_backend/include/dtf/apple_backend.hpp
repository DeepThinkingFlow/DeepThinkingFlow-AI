#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace dtf {

struct AppleBuildConfig {
  bool apple_silicon_enabled;
  bool mlx_enabled;
  bool coreml_enabled;
  bool strict_mode_enabled;
};

struct AppleDeviceInfo {
  std::string chip_family;
  std::size_t unified_memory_bytes;
  bool metal_available;
  bool coreml_available;
};

struct AppleRuntimeOptions {
  bool use_mps_graph;
  bool use_binary_archive_cache;
  bool prefer_unified_memory_views;
  bool prefer_coreml_for_ane;
};

struct AppleTelemetry {
  bool binary_archive_expected;
  bool unified_memory_assumed;
  bool command_queue_required;
  bool mps_graph_preferred;
  bool runtime_options_accepted;
  std::string notes;
};

class AppleBackend {
 public:
  AppleBackend();
  ~AppleBackend();

  AppleBackend(const AppleBackend&) = delete;
  AppleBackend& operator=(const AppleBackend&) = delete;

  AppleBuildConfig build_config() const;
  std::vector<AppleDeviceInfo> enumerate_devices() const;
  AppleRuntimeOptions default_runtime_options() const;
  AppleTelemetry configure_runtime(const AppleRuntimeOptions& options);
};

}  // namespace dtf
