#include "dtf/apple_backend.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(_dtf_apple_backend, m) {
  m.doc() = "DeepThinkingFlow Apple Silicon backend bindings";

  py::class_<dtf::AppleBuildConfig>(m, "AppleBuildConfig")
      .def_readonly("apple_silicon_enabled", &dtf::AppleBuildConfig::apple_silicon_enabled)
      .def_readonly("mlx_enabled", &dtf::AppleBuildConfig::mlx_enabled)
      .def_readonly("coreml_enabled", &dtf::AppleBuildConfig::coreml_enabled)
      .def_readonly("strict_mode_enabled", &dtf::AppleBuildConfig::strict_mode_enabled);

  py::class_<dtf::AppleDeviceInfo>(m, "AppleDeviceInfo")
      .def_readonly("chip_family", &dtf::AppleDeviceInfo::chip_family)
      .def_readonly("unified_memory_bytes", &dtf::AppleDeviceInfo::unified_memory_bytes)
      .def_readonly("metal_available", &dtf::AppleDeviceInfo::metal_available)
      .def_readonly("coreml_available", &dtf::AppleDeviceInfo::coreml_available);

  py::class_<dtf::AppleRuntimeOptions>(m, "AppleRuntimeOptions")
      .def(py::init<>())
      .def_readwrite("use_mps_graph", &dtf::AppleRuntimeOptions::use_mps_graph)
      .def_readwrite("use_binary_archive_cache", &dtf::AppleRuntimeOptions::use_binary_archive_cache)
      .def_readwrite("prefer_unified_memory_views", &dtf::AppleRuntimeOptions::prefer_unified_memory_views)
      .def_readwrite("prefer_coreml_for_ane", &dtf::AppleRuntimeOptions::prefer_coreml_for_ane);

  py::class_<dtf::AppleTelemetry>(m, "AppleTelemetry")
      .def_readonly("binary_archive_expected", &dtf::AppleTelemetry::binary_archive_expected)
      .def_readonly("unified_memory_assumed", &dtf::AppleTelemetry::unified_memory_assumed)
      .def_readonly("command_queue_required", &dtf::AppleTelemetry::command_queue_required)
      .def_readonly("mps_graph_preferred", &dtf::AppleTelemetry::mps_graph_preferred)
      .def_readonly("runtime_options_accepted", &dtf::AppleTelemetry::runtime_options_accepted)
      .def_readonly("notes", &dtf::AppleTelemetry::notes);

  py::class_<dtf::AppleBackend>(m, "AppleBackend")
      .def(py::init<>())
      .def("build_config", &dtf::AppleBackend::build_config)
      .def("enumerate_devices", &dtf::AppleBackend::enumerate_devices)
      .def("default_runtime_options", &dtf::AppleBackend::default_runtime_options)
      .def("configure_runtime", &dtf::AppleBackend::configure_runtime, py::arg("options"));
}
