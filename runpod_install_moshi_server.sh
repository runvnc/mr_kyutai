#!/usr/bin/env bash
set -e

export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}
export LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:${LIBRARY_PATH}
export CPATH=/usr/local/cuda/targets/x86_64-linux/include:${CPATH}
export RUSTFLAGS="-C linker=gcc -L /usr/local/cuda/targets/x86_64-linux/lib"

echo "CUDA_HOME=$CUDA_HOME"
ls -la /usr/local/cuda/targets/x86_64-linux/lib | egrep "cublas|cublasLt|curand|nvrtc" || true

cargo install --features cuda moshi-server
