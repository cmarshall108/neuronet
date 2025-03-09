#!/usr/bin/env bash

# Exit script if any command fails
set -e

# Create build directory if it doesn't exist
mkdir -p build
cd build

# Check for CMAKE_CUDA_ARCHITECTURES override
CUDA_ARCH=${CMAKE_CUDA_ARCHITECTURES:-"37"}

# Configure the build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DNEURONET_USE_CUDA=${NEURONET_USE_CUDA:-ON} \
  -DNEURONET_USE_METAL=${NEURONET_USE_METAL:-ON} \
  -DNEURONET_OPTIMIZE_K80=ON \
  -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}

# Build the library and examples
cmake --build . -- -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Run tests if requested
if [ "${RUN_TESTS}" == "1" ]; then
  echo "Running tests..."
  ctest --output-on-failure
fi

# Install if requested
if [ "${INSTALL_NEURONET}" == "1" ]; then
  echo "Installing NeuroNet..."
  sudo cmake --install .
fi

echo "NeuroNet build completed successfully!"
echo "Examples can be found in: $(pwd)/bin"

# Print hardware detection info
if [ -f bin/benchmark ]; then
  echo -e "\nHardware detection:"
  bin/benchmark --info-only
fi
