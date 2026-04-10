#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

DIST_DIR="${DIST_DIR:-dist}"
OPENBLAS_PREFIX="${OPENBLAS_PREFIX:-$(brew --prefix openblas)}"
LIBOMP_PREFIX="${LIBOMP_PREFIX:-$(brew --prefix libomp)}"
export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-${OPENBLAS_PREFIX};${LIBOMP_PREFIX}}"
export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-15.0}"

mkdir -p "${DIST_DIR}"

uv build --wheel . \
  --out-dir "${DIST_DIR}/" \
  -Ccmake.define.FAISS_ENABLE_METAL=ON \
  -Ccmake.define.FAISS_ENABLE_GPU=OFF \
  -Ccmake.define.FAISS_ENABLE_C_API=ON \
  -Ccmake.define.FAISS_ENABLE_MKL=OFF \
  -Ccmake.define.BUILD_SHARED_LIBS=ON \
  -Ccmake.define.BUILD_TESTING=OFF \
  -Ccmake.define.FAISS_ENABLE_EXTRAS=OFF \
  -Ccmake.define.OpenMP_C_FLAGS="-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include" \
  -Ccmake.define.OpenMP_C_LIB_NAMES="omp" \
  -Ccmake.define.OpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include" \
  -Ccmake.define.OpenMP_CXX_LIB_NAMES="omp" \
  -Ccmake.define.OpenMP_omp_LIBRARY="${LIBOMP_PREFIX}/lib/libomp.dylib"

"${SCRIPT_DIR}/repair-metal-wheel-osx-arm64.sh"

ls -lh "${DIST_DIR}"/*.whl
