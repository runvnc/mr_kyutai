#!/bin/bash
set -ex
cd "$(dirname "$0")/"

# This is part of a hack to get dependencies needed for the TTS Rust server, because it integrates a Python component
[ -f pyproject.toml ] || wget https://raw.githubusercontent.com/kyutai-labs/moshi/9837ca328d58deef5d7a4fe95a0fb49c902ec0ae/rust/moshi-server/pyproject.toml
[ -f uv.lock ] || wget https://raw.githubusercontent.com/kyutai-labs/moshi/9837ca328d58deef5d7a4fe95a0fb49c902ec0ae/rust/moshi-server/uv.lock

uv venv
source .venv/bin/activate

cd ..

# Find CUDA libraries first
echo "=== Finding CUDA libraries ==="
CUDA_LIB_DIRS=""
for lib in libnvrtc.so libcurand.so libcublas.so libcublasLt.so; do
    LIB_PATH=$(find /usr /opt -name "${lib}*" 2>/dev/null | head -1)
    if [ -n "$LIB_PATH" ]; then
        DIR=$(dirname "$LIB_PATH")
        echo "Found $lib in: $DIR"
        if [[ ! "$CUDA_LIB_DIRS" =~ "$DIR" ]]; then
            CUDA_LIB_DIRS="$CUDA_LIB_DIRS:$DIR"
        fi
    fi
done
CUDA_LIB_DIRS="${CUDA_LIB_DIRS#:}"

# Get Python library path
PYTHON_LIB_DIR=$(python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')

# Combine both paths - CUDA first, then Python
export LD_LIBRARY_PATH="${CUDA_LIB_DIRS}:${PYTHON_LIB_DIR}:${LD_LIBRARY_PATH}"
export LIBRARY_PATH="${CUDA_LIB_DIRS}:${LIBRARY_PATH}"

echo "=== Library paths set ==="
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "LIBRARY_PATH: $LIBRARY_PATH"
echo ""

# Find CUDA root
CUDA_ROOT=$(find /usr/local -maxdepth 1 -name "cuda*" -type d 2>/dev/null | head -1)
if [ -z "$CUDA_ROOT" ]; then
    CUDA_ROOT="/usr/local/cuda"
fi
export CUDA_PATH="$CUDA_ROOT"

# A fix for building Sentencepiece on GCC 15, see: https://github.com/google/sentencepiece/issues/1108
export CXXFLAGS="-include cstdint"

# If you already have moshi-server installed and things are not working because of the LD_LIBRARY_PATH issue,
# you might have to force a rebuild with --force.
cargo install --features cuda moshi-server@0.6.4

# If you're getting `moshi-server: error: unrecognized arguments: worker`, it means you're
# using the binary from the `moshi` Python package rather than from the Rust package.
# Use `pip install moshi --upgrade` to update the Python package to >=0.2.8.
uv run --locked --project ./dockerless moshi-server worker --config services/moshi-server/configs/tts.toml --port 8089
