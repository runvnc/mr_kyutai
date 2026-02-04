#!/bin/bash
set -e

echo "=== Creating CUDA library symlinks ==="
echo ""

# Source locations (in Python dist-packages)
NVRTC_SRC="/usr/local/lib/python3.12/dist-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12"
CURAND_SRC="/usr/local/lib/python3.12/dist-packages/nvidia/curand/lib/libcurand.so.10"
CUBLAS_SRC="/usr/local/lib/python3.12/dist-packages/nvidia/cublas/lib/libcublas.so.12"
CUBLASLT_SRC="/usr/local/lib/python3.12/dist-packages/nvidia/cublas/lib/libcublasLt.so.12"

# Target directory (where linker looks)
TARGET_DIR="/usr/local/cuda-12.8/lib64"

# Create target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Create symlinks without version numbers (what the linker wants)
echo "Creating symlink: libnvrtc.so -> $NVRTC_SRC"
ln -sf "$NVRTC_SRC" "$TARGET_DIR/libnvrtc.so"

echo "Creating symlink: libcurand.so -> $CURAND_SRC"
ln -sf "$CURAND_SRC" "$TARGET_DIR/libcurand.so"

echo "Creating symlink: libcublas.so -> $CUBLAS_SRC"
ln -sf "$CUBLAS_SRC" "$TARGET_DIR/libcublas.so"

echo "Creating symlink: libcublasLt.so -> $CUBLASLT_SRC"
ln -sf "$CUBLASLT_SRC" "$TARGET_DIR/libcublasLt.so"

echo ""
echo "=== Verifying symlinks ==="
ls -lh "$TARGET_DIR"/lib{nvrtc,curand,cublas,cublasLt}.so

echo ""
echo "Done! Now you can run the build script."
