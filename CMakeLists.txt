cmake_minimum_required(VERSION 3.18)
project(neuronet VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Options
option(NEURONET_USE_CUDA "Build with CUDA support" ON)
option(NEURONET_USE_METAL "Build with Metal support on macOS" ON)
option(NEURONET_BUILD_TESTS "Build tests" ON)
option(NEURONET_BUILD_EXAMPLES "Build examples" ON)

# Find required packages
find_package(Threads REQUIRED)
find_package(CURL REQUIRED)

# Check for macOS to enable Metal
if(APPLE AND NEURONET_USE_METAL)
  enable_language(OBJC)
  enable_language(OBJCXX)
  set(CMAKE_OBJC_FLAGS "${CMAKE_OBJC_FLAGS} -fobjc-arc")
  set(CMAKE_OBJCXX_FLAGS "${CMAKE_OBJCXX_FLAGS} -fobjc-arc")
  add_definitions(-DNEURONET_USE_METAL=1)
endif()

# Check for CUDA
if(NEURONET_USE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    add_definitions(-DNEURONET_USE_CUDA=1)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_ARCHITECTURES 35 50 60 70)  # Tesla K80 is compute capability 3.7
  else()
    message(WARNING "CUDA requested but not found, building without CUDA support")
    set(NEURONET_USE_CUDA OFF)
  endif()
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Core library source files
set(CORE_SOURCES
    src/core/tensor.cpp
    src/core/device.cpp
    src/core/memory.cpp
    src/core/ops.cpp
    src/nn/layers.cpp
    src/nn/activations.cpp
    src/nn/loss.cpp
    src/nn/optimizer.cpp
    src/models/model.cpp
    src/models/huggingface.cpp
    src/utils/logging.cpp
    src/utils/json.cpp
)

# Add CPU backend sources
set(CPU_SOURCES
    src/backends/cpu/cpu_backend.cpp
    src/backends/cpu/cpu_ops.cpp
)

# Add Metal backend sources if enabled
if(APPLE AND NEURONET_USE_METAL)
  set(METAL_SOURCES
      src/backends/metal/metal_backend.mm
      src/backends/metal/metal_ops.mm
  )
endif()

# Add CUDA backend sources if enabled
if(NEURONET_USE_CUDA)
  set(CUDA_SOURCES
      src/backends/cuda/cuda_backend.cu
      src/backends/cuda/cuda_ops.cu
  )
endif()

# Combine all sources
set(NEURONET_SOURCES
    ${CORE_SOURCES}
    ${CPU_SOURCES}
    ${METAL_SOURCES}
    ${CUDA_SOURCES}
)

# Create the main library
add_library(neuronet ${NEURONET_SOURCES})

# Link libraries
target_link_libraries(neuronet PRIVATE Threads::Threads CURL::libcurl)

if(APPLE AND NEURONET_USE_METAL)
  target_link_libraries(neuronet PRIVATE "-framework Foundation" "-framework Metal" "-framework MetalPerformanceShaders")
endif()

# Install rules
install(TARGETS neuronet
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)

install(DIRECTORY include/ DESTINATION include)

# Build examples
if(NEURONET_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

# Build tests
if(NEURONET_BUILD_TESTS)
  add_subdirectory(tests)
endif()
