#!/bin/bash

echo "=== CUDA Library Path Setup ==="
echo ""

# Find CUDA libraries
echo "Searching for CUDA libraries..."
NVRTC_PATH=$(find /usr /opt -name "libnvrtc.so*" 2>/dev/null | head -1)
CURAND_PATH=$(find /usr /opt -name "libcurand.so*" 2>/dev/null | head -1)
CUBLAS_PATH=$(find /usr /opt -name "libcublas.so*" 2>/dev/null | head -1)
CUBLASLT_PATH=$(find /usr /opt -name "libcublasLt.so*" 2>/dev/null | head -1)

# Extract directory paths
CUDA_LIB_DIRS=""

if [ -n "$NVRTC_PATH" ]; then
    DIR=$(dirname "$NVRTC_PATH")
    echo "Found libnvrtc in: $DIR"
    CUDA_LIB_DIRS="$DIR"
fi

if [ -n "$CURAND_PATH" ]; then
    DIR=$(dirname "$CURAND_PATH")
    echo "Found libcurand in: $DIR"
    if [[ ! "$CUDA_LIB_DIRS" =~ "$DIR" ]]; then
        CUDA_LIB_DIRS="$CUDA_LIB_DIRS:$DIR"
    fi
fi

if [ -n "$CUBLAS_PATH" ]; then
    DIR=$(dirname "$CUBLAS_PATH")
    echo "Found libcublas in: $DIR"
    if [[ ! "$CUDA_LIB_DIRS" =~ "$DIR" ]]; then
        CUDA_LIB_DIRS="$CUDA_LIB_DIRS:$DIR"
    fi
fi

if [ -n "$CUBLASLT_PATH" ]; then
    DIR=$(dirname "$CUBLASLT_PATH")
    echo "Found libcublasLt in: $DIR"
    if [[ ! "$CUDA_LIB_DIRS" =~ "$DIR" ]]; then
        CUDA_LIB_DIRS="$CUDA_LIB_DIRS:$DIR"
    fi
fi

# Remove leading colon if present
CUDA_LIB_DIRS="${CUDA_LIB_DIRS#:}"

if [ -z "$CUDA_LIB_DIRS" ]; then
    echo ""
    echo "ERROR: Could not find CUDA libraries!"
    echo "Make sure CUDA is installed."
    exit 1
fi

echo ""
echo "=== Setting Environment Variables ==="
echo ""

# Find CUDA installation root
CUDA_ROOT=$(find /usr/local -maxdepth 1 -name "cuda*" -type d 2>/dev/null | head -1)
if [ -z "$CUDA_ROOT" ]; then
    CUDA_ROOT="/usr/local/cuda"
fi

echo "export CUDA_PATH=$CUDA_ROOT"
echo "export LD_LIBRARY_PATH=$CUDA_LIB_DIRS:\$LD_LIBRARY_PATH"
echo "export LIBRARY_PATH=$CUDA_LIB_DIRS:\$LIBRARY_PATH"
echo ""

# Actually export them
export CUDA_PATH="$CUDA_ROOT"
export LD_LIBRARY_PATH="$CUDA_LIB_DIRS:$LD_LIBRARY_PATH"
export LIBRARY_PATH="$CUDA_LIB_DIRS:$LIBRARY_PATH"

echo "Environment variables set!"
echo ""
echo "To use in your current shell, run:"
echo "  source /tmp/setup_cuda_env.sh"
echo ""
echo "Or add these lines to your ~/.bashrc:"
echo "  export CUDA_PATH=$CUDA_ROOT"
echo "  export LD_LIBRARY_PATH=$CUDA_LIB_DIRS:\$LD_LIBRARY_PATH"
echo "  export LIBRARY_PATH=$CUDA_LIB_DIRS:\$LIBRARY_PATH"
