cmake_minimum_required(VERSION 3.26)

project(DeepThinkingFlowCudaBackend LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(DTF_CUDA_ENABLED "Build with CUDA" OFF)
option(DTF_CUDA_STRICT "Fail fast on risky CUDA configuration choices." ON)
option(DTF_ENABLE_FLASH_ATTENTION "Enable FlashAttention 2 integration when available." ON)
option(DTF_ENABLE_BITSANDBYTES "Enable bitsandbytes-oriented integration hooks." ON)
option(DTF_ENABLE_NCCL "Enable NCCL-oriented integration hooks." ON)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

if(DTF_CUDA_ENABLED)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    message(FATAL_ERROR "CMAKE_CUDA_ARCHITECTURES is empty. Set it to your exact NVIDIA GPU SM, for example 89 or 90. Do not use a generic arch if you care about performance.")
  endif()
  if(DTF_CUDA_STRICT)
    if(CMAKE_CUDA_ARCHITECTURES MATCHES "[,; ]")
      message(FATAL_ERROR "CMAKE_CUDA_ARCHITECTURES must be a single exact SM target in strict mode.")
    endif()
    if(CMAKE_CUDA_ARCHITECTURES STREQUAL "native")
      message(FATAL_ERROR "Do not pass native as CMAKE_CUDA_ARCHITECTURES. Use the exact SM number instead.")
    endif()
  endif()
  find_package(CUDAToolkit REQUIRED)
endif()

set(DTF_CUDA_SOURCES
  src/python_bindings.cpp
  src/cuda_backend.cpp
)

if(DTF_CUDA_ENABLED)
  list(APPEND DTF_CUDA_SOURCES src/cuda_runtime.cu)
else()
  list(APPEND DTF_CUDA_SOURCES src/cpu_runtime_stub.cpp)
endif()

pybind11_add_module(_dtf_cuda_backend MODULE ${DTF_CUDA_SOURCES})

target_include_directories(
  _dtf_cuda_backend
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(DTF_CUDA_ENABLED)
  target_link_libraries(
    _dtf_cuda_backend
    PRIVATE
      CUDA::cudart
      CUDA::cublas
      CUDA::cublasLt
  )
endif()

target_compile_features(_dtf_cuda_backend PRIVATE cxx_std_20)

target_compile_definitions(
  _dtf_cuda_backend
  PRIVATE
    DTF_CUDA_ENABLED=$<BOOL:${DTF_CUDA_ENABLED}>
    DTF_CUDA_STRICT=$<BOOL:${DTF_CUDA_STRICT}>
    DTF_ENABLE_FLASH_ATTENTION=$<BOOL:${DTF_ENABLE_FLASH_ATTENTION}>
    DTF_ENABLE_BITSANDBYTES=$<BOOL:${DTF_ENABLE_BITSANDBYTES}>
    DTF_ENABLE_NCCL=$<BOOL:${DTF_ENABLE_NCCL}>
)

target_compile_options(
  _dtf_cuda_backend
  PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -fPIC>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --use_fast_math -lineinfo>
)

if(DTF_CUDA_ENABLED)
  set_source_files_properties(
    src/cuda_runtime.cu
    PROPERTIES
      LANGUAGE CUDA
  )
endif()

install(TARGETS _dtf_cuda_backend LIBRARY DESTINATION deepthinkingflow_cuda)
