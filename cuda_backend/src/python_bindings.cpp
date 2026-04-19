#include "dtf/cuda_backend.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(_dtf_cuda_backend, m) {
  m.doc() = "DeepThinkingFlow CUDA backend bindings";

  py::class_<dtf::BuildConfig>(m, "BuildConfig")
      .def_readonly("cuda_architectures", &dtf::BuildConfig::cuda_architectures)
      .def_readonly("flash_attention_enabled", &dtf::BuildConfig::flash_attention_enabled)
      .def_readonly("bitsandbytes_enabled", &dtf::BuildConfig::bitsandbytes_enabled)
      .def_readonly("nccl_enabled", &dtf::BuildConfig::nccl_enabled);

  py::class_<dtf::DeviceInfo>(m, "DeviceInfo")
      .def_readonly("device_index", &dtf::DeviceInfo::device_index)
      .def_readonly("name", &dtf::DeviceInfo::name)
      .def_readonly("major_cc", &dtf::DeviceInfo::major_cc)
      .def_readonly("minor_cc", &dtf::DeviceInfo::minor_cc)
      .def_readonly("total_global_mem_bytes", &dtf::DeviceInfo::total_global_mem_bytes)
      .def_readonly("multiprocessor_count", &dtf::DeviceInfo::multiprocessor_count);

  py::class_<dtf::RuntimeOptions>(m, "RuntimeOptions")
      .def(py::init<>())
      .def_readwrite("compute_stream_priority", &dtf::RuntimeOptions::compute_stream_priority)
      .def_readwrite("memcpy_stream_priority", &dtf::RuntimeOptions::memcpy_stream_priority)
      .def_readwrite("enable_async_pool", &dtf::RuntimeOptions::enable_async_pool)
      .def_readwrite(
          "memory_pool_release_threshold_bytes",
          &dtf::RuntimeOptions::memory_pool_release_threshold_bytes);

  py::class_<dtf::EngineTelemetry>(m, "EngineTelemetry")
      .def_readonly("async_allocator_available", &dtf::EngineTelemetry::async_allocator_available)
      .def_readonly("memory_pool_enabled", &dtf::EngineTelemetry::memory_pool_enabled)
      .def_readonly("separate_streams_configured", &dtf::EngineTelemetry::separate_streams_configured)
      .def_readonly(
          "profiling_required_before_optimization",
          &dtf::EngineTelemetry::profiling_required_before_optimization)
      .def_readonly("notes", &dtf::EngineTelemetry::notes);

  py::class_<dtf::CudaBackend>(m, "CudaBackend")
      .def(py::init<>())
      .def("build_config", &dtf::CudaBackend::build_config)
      .def("enumerate_devices", &dtf::CudaBackend::enumerate_devices)
      .def("default_runtime_options", &dtf::CudaBackend::default_runtime_options)
      .def("configure_runtime", &dtf::CudaBackend::configure_runtime, py::arg("options"));
}
