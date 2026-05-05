#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

DIST_DIR="${DIST_DIR:-dist}"
CONDA_ENV_PREFIX="${CONDA_PREFIX:-}"

if [[ -z "${CONDA_ENV_PREFIX}" ]]; then
  echo "CONDA_PREFIX is not set. Activate the conda environment before building." >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV_PREFIX}/bin/python}"
UV_BIN="${UV_BIN:-${CONDA_ENV_PREFIX}/bin/uv}"
OPENBLAS_PREFIX="${OPENBLAS_PREFIX:-${CONDA_ENV_PREFIX}}"
LIBOMP_PREFIX="${LIBOMP_PREFIX:-${CONDA_ENV_PREFIX}}"
LIBOMP_INCLUDE_DIR="${LIBOMP_INCLUDE_DIR:-${LIBOMP_PREFIX}/include}"
LIBOMP_DYLIB="${LIBOMP_DYLIB:-${LIBOMP_PREFIX}/lib/libomp.dylib}"
OPENBLAS_DYLIB="${OPENBLAS_DYLIB:-${OPENBLAS_PREFIX}/lib/libopenblas.dylib}"
LAPACK_DYLIB="${LAPACK_DYLIB:-${OPENBLAS_PREFIX}/lib/liblapack.dylib}"
OPENMP_FLAGS="-Xpreprocessor -fopenmp"

if [[ ! -f "${LIBOMP_INCLUDE_DIR}/omp.h" ]]; then
  echo "omp.h not found: ${LIBOMP_INCLUDE_DIR}/omp.h" >&2
  echo "Install libomp into the active conda environment or set LIBOMP_PREFIX." >&2
  exit 1
fi

if [[ ! -f "${LIBOMP_DYLIB}" ]]; then
  echo "libomp dylib not found: ${LIBOMP_DYLIB}" >&2
  echo "Install libomp into the active conda environment or set LIBOMP_PREFIX." >&2
  exit 1
fi

if [[ ! -f "${OPENBLAS_DYLIB}" ]]; then
  echo "OpenBLAS dylib not found: ${OPENBLAS_DYLIB}" >&2
  echo "Install openblas into the active conda environment or set OPENBLAS_PREFIX/OPENBLAS_DYLIB." >&2
  exit 1
fi

if [[ ! -f "${LAPACK_DYLIB}" ]]; then
  echo "LAPACK dylib not found: ${LAPACK_DYLIB}" >&2
  echo "Install lapack into the active conda environment or set LAPACK_DYLIB." >&2
  exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not found in active conda environment: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -x "${UV_BIN}" ]]; then
  echo "uv not found in active conda environment: ${UV_BIN}" >&2
  echo "Install uv into the active conda environment with ./scripts/ci/install-metal-deps-osx-arm64.sh." >&2
  exit 1
fi

export PATH="${CONDA_ENV_PREFIX}/bin:${PATH}"

if [[ -n "${CMAKE_PREFIX_PATH:-}" ]]; then
  export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH};${CONDA_ENV_PREFIX};${OPENBLAS_PREFIX};${LIBOMP_PREFIX}"
else
  export CMAKE_PREFIX_PATH="${CONDA_ENV_PREFIX};${OPENBLAS_PREFIX};${LIBOMP_PREFIX}"
fi

export OpenMP_ROOT="${OpenMP_ROOT:-${LIBOMP_PREFIX}}"
export CPPFLAGS="${CPPFLAGS:-} -I${LIBOMP_INCLUDE_DIR}"
export CFLAGS="${CFLAGS:-} -I${LIBOMP_INCLUDE_DIR}"
export CXXFLAGS="${CXXFLAGS:-} -I${LIBOMP_INCLUDE_DIR}"
export LDFLAGS="${LDFLAGS:-} -L${LIBOMP_PREFIX}/lib -L${OPENBLAS_PREFIX}/lib"
export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-15.0}"

mkdir -p "${DIST_DIR}"

"${UV_BIN}" build --wheel --no-build-isolation --python "${PYTHON_BIN}" . \
  --out-dir "${DIST_DIR}/" \
  -Ccmake.define.FAISS_ENABLE_METAL=ON \
  -Ccmake.define.FAISS_ENABLE_GPU=OFF \
  -Ccmake.define.FAISS_ENABLE_C_API=ON \
  -Ccmake.define.FAISS_ENABLE_MKL=OFF \
  -Ccmake.define.BLA_VENDOR=OpenBLAS \
  -Ccmake.define.BLAS_LIBRARIES="${OPENBLAS_DYLIB}" \
  -Ccmake.define.LAPACK_LIBRARIES="${LAPACK_DYLIB}" \
  -Ccmake.define.BUILD_SHARED_LIBS=ON \
  -Ccmake.define.BUILD_TESTING=OFF \
  -Ccmake.define.FAISS_ENABLE_EXTRAS=OFF \
  -Ccmake.define.OpenMP_ROOT="${OpenMP_ROOT}" \
  -Ccmake.define.OpenMP_C_FLAGS="${OPENMP_FLAGS}" \
  -Ccmake.define.OpenMP_C_INCLUDE_DIR="${LIBOMP_INCLUDE_DIR}" \
  -Ccmake.define.OpenMP_C_LIB_NAMES="omp" \
  -Ccmake.define.OpenMP_CXX_FLAGS="${OPENMP_FLAGS}" \
  -Ccmake.define.OpenMP_CXX_INCLUDE_DIR="${LIBOMP_INCLUDE_DIR}" \
  -Ccmake.define.OpenMP_CXX_LIB_NAMES="omp" \
  -Ccmake.define.OpenMP_omp_LIBRARY="${LIBOMP_DYLIB}"

"${SCRIPT_DIR}/repair-metal-wheel-osx-arm64.sh"

ls -lh "${DIST_DIR}"/*.whl
