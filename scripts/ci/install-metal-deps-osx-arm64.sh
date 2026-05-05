#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "CONDA_PREFIX is not set. Activate the conda environment before installing dependencies." >&2
  exit 1
fi

conda install -y -c conda-forge \
  cmake \
  swig \
  llvm-openmp \
  openblas \
  lapack \
  numpy \
  packaging \
  scikit-build-core \
  uv
