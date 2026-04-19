cmake_minimum_required(VERSION 3.26)

project(DeepThinkingFlowAppleBackend LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 20)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

option(DTF_APPLE_SILICON_ENABLED "Build Apple Silicon backend with Metal/MPS/Core ML integration hooks." OFF)
option(DTF_APPLE_STRICT "Fail fast on risky Apple Silicon configuration choices." ON)
option(DTF_ENABLE_MLX "Enable MLX-oriented integration hooks." ON)
option(DTF_ENABLE_COREML "Enable Core ML / ANE-oriented integration hooks." ON)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

set(DTF_APPLE_SOURCES
  src/python_bindings.mm
  src/apple_backend.cpp
)

if(DTF_APPLE_SILICON_ENABLED)
  if(NOT APPLE)
    message(FATAL_ERROR "DTF_APPLE_SILICON_ENABLED requires building on Apple platforms.")
  endif()
  if(DTF_APPLE_STRICT AND NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
    message(FATAL_ERROR "Strict Apple mode requires arm64 / Apple Silicon target.")
  endif()
  list(APPEND DTF_APPLE_SOURCES src/metal_runtime.mm)
else()
  list(APPEND DTF_APPLE_SOURCES src/cpu_runtime_stub.cpp)
endif()

pybind11_add_module(_dtf_apple_backend MODULE ${DTF_APPLE_SOURCES})

target_include_directories(
  _dtf_apple_backend
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(_dtf_apple_backend PRIVATE cxx_std_20)

target_compile_definitions(
  _dtf_apple_backend
  PRIVATE
    DTF_APPLE_SILICON_ENABLED=$<BOOL:${DTF_APPLE_SILICON_ENABLED}>
    DTF_APPLE_STRICT=$<BOOL:${DTF_APPLE_STRICT}>
    DTF_ENABLE_MLX=$<BOOL:${DTF_ENABLE_MLX}>
    DTF_ENABLE_COREML=$<BOOL:${DTF_ENABLE_COREML}>
)

target_compile_options(
  _dtf_apple_backend
  PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -fPIC>
    $<$<COMPILE_LANGUAGE:OBJCXX>:-O3 -fobjc-arc>
)

if(DTF_APPLE_SILICON_ENABLED)
  find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
  find_library(METAL_FRAMEWORK Metal REQUIRED)
  find_library(MPS_FRAMEWORK MetalPerformanceShaders REQUIRED)
  find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
  find_library(COREML_FRAMEWORK CoreML REQUIRED)

  target_link_libraries(
    _dtf_apple_backend
    PRIVATE
      ${FOUNDATION_FRAMEWORK}
      ${METAL_FRAMEWORK}
      ${MPS_FRAMEWORK}
      ${ACCELERATE_FRAMEWORK}
      ${COREML_FRAMEWORK}
  )
endif()

install(TARGETS _dtf_apple_backend LIBRARY DESTINATION deepthinkingflow_apple)
