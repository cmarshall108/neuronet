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
option(NEURONET_OPTIMIZE_K80 "Specifically optimize for Tesla K80 GPU" ON)

# Find required packages
find_package(Threads REQUIRED)
find_package(CURL REQUIRED)

# nlohmann/json is required
include(FetchContent)
FetchContent_Declare(
  json
  URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz
)
FetchContent_MakeAvailable(json)

# Check for macOS to enable Metal
if(APPLE AND NEURONET_USE_METAL)
  enable_language(OBJC)
  enable_language(OBJCXX)
  set(CMAKE_OBJC_FLAGS "${CMAKE_OBJC_FLAGS} -fobjc-arc")
  set(CMAKE_OBJCXX_FLAGS "${CMAKE_OBJCXX_FLAGS} -fobjc-arc")
  add_definitions(-DNEURONET_USE_METAL=1)
  
  # Check for Apple Silicon - safer detection method
  execute_process(
    COMMAND sysctl -n machdep.cpu.brand_string
    OUTPUT_VARIABLE APPLE_CPU_TYPE
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  
  # Also check the architecture
  execute_process(
    COMMAND uname -m
    OUTPUT_VARIABLE ARCH_TYPE
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  
  # Only set Apple Silicon if architecture is arm64 and CPU brand contains "Apple"
  if(ARCH_TYPE STREQUAL "arm64" AND APPLE_CPU_TYPE MATCHES "Apple")
    message(STATUS "Apple Silicon detected: ${APPLE_CPU_TYPE}")
    add_definitions(-DNEURONET_APPLE_SILICON=1)
    set(IS_APPLE_SILICON TRUE)
  else()
    message(STATUS "Intel Mac detected (or you are running with `arch` set): ${APPLE_CPU_TYPE}")
    set(IS_APPLE_SILICON FALSE)
  endif()
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
    
    # Tesla K80 is compute capability 3.7
    if(NEURONET_OPTIMIZE_K80)
      set(CMAKE_CUDA_ARCHITECTURES 37)
      message(STATUS "Optimizing for Tesla K80 (compute capability 3.7)")
    else()
      # Support multiple architectures
      set(CMAKE_CUDA_ARCHITECTURES 35 37 50 60 70)
    endif()
    
    # Find CUDA libraries
    find_library(CUDA_CUDART_LIBRARY cudart PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    find_library(CUDA_CUBLAS_LIBRARY cublas PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    
    # Check if Tesla K80 is available in the system
    execute_process(COMMAND nvidia-smi -L
                    OUTPUT_VARIABLE NVIDIA_GPU_INFO
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET)
    if(NVIDIA_GPU_INFO MATCHES "Tesla K80")
      message(STATUS "Tesla K80 GPU detected")
      add_definitions(-DNEURONET_TESLA_K80_AVAILABLE=1)
    endif()
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
      src/backends/metal/metal_wrapper.mm
  )
  
  # Make sure we compile these as Objective-C++
  set_source_files_properties(
      ${METAL_SOURCES}
      PROPERTIES
      COMPILE_FLAGS "-x objective-c++"
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
target_link_libraries(neuronet PRIVATE 
    Threads::Threads 
    CURL::libcurl
    nlohmann_json::nlohmann_json)

# Add compiler optimization flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(neuronet PRIVATE -O3 -march=native)
    # Only add -mcpu=apple-m1 for Apple Silicon
    if(IS_APPLE_SILICON)
        target_compile_options(neuronet PRIVATE -mcpu=apple-m1)
    endif()
endif()

if(NEURONET_USE_CUDA)
  target_link_libraries(neuronet PRIVATE ${CUDA_CUDART_LIBRARY} ${CUDA_CUBLAS_LIBRARY})
endif()

if(APPLE AND NEURONET_USE_METAL)
  target_link_libraries(neuronet PRIVATE "-framework Foundation" "-framework Metal" "-framework MetalPerformanceShaders")
endif()

# Add version to preprocessor defines
target_compile_definitions(neuronet PRIVATE
    NEURONET_VERSION="${PROJECT_VERSION}"
    NEURONET_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
    NEURONET_VERSION_MINOR=${PROJECT_VERSION_MINOR}
    NEURONET_VERSION_PATCH=${PROJECT_VERSION_PATCH}
)

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

# Add packaging supportinclude(CPack)
set(CPACK_PACKAGE_NAME "neuronet")
set(CPACK_PACKAGE_VENDOR "NeuroNet")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Tensor computation library optimized for K80 and Metal")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
