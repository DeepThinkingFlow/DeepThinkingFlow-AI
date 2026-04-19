#include "dtf/apple_backend.hpp"

namespace dtf {

AppleBackend::AppleBackend() = default;
AppleBackend::~AppleBackend() = default;

AppleBuildConfig AppleBackend::build_config() const {
  return AppleBuildConfig{
#if DTF_APPLE_SILICON_ENABLED
      .apple_silicon_enabled = true,
#else
      .apple_silicon_enabled = false,
#endif
#if DTF_ENABLE_MLX
      .mlx_enabled = true,
#else
      .mlx_enabled = false,
#endif
#if DTF_ENABLE_COREML
      .coreml_enabled = true,
#else
      .coreml_enabled = false,
#endif
#if DTF_APPLE_STRICT
      .strict_mode_enabled = true,
#else
      .strict_mode_enabled = false,
#endif
  };
}

AppleRuntimeOptions AppleBackend::default_runtime_options() const {
  return AppleRuntimeOptions{
      .use_mps_graph = true,
      .use_binary_archive_cache = true,
      .prefer_unified_memory_views = true,
      .prefer_coreml_for_ane = true,
  };
}

}  // namespace dtf
