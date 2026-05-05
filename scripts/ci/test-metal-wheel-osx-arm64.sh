#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

DIST_DIR="${DIST_DIR:-dist}"
VENV_DIR="${VENV_DIR:-.venv-test}"
WHEEL_PATH="${WHEEL_PATH:-}"

if [ -z "${WHEEL_PATH}" ]; then
  wheels=()
  while IFS= read -r wheel; do
    wheels+=("${wheel}")
  done < <(find "${DIST_DIR}" -maxdepth 1 -type f -name '*.whl' -exec ls -1t {} +)
  if [ "${#wheels[@]}" -eq 0 ]; then
    echo "No wheels found in ${DIST_DIR}" >&2
    exit 1
  fi
  WHEEL_PATH="${wheels[0]}"
fi

mkdir -p test-results/pytest

uv venv "${VENV_DIR}"
echo "Using wheel: ${WHEEL_PATH}"
uv pip install --python "${VENV_DIR}/bin/python" "${WHEEL_PATH}" numpy scipy

"${VENV_DIR}/bin/python" -X faulthandler - <<'PY'
import faiss
import numpy as np

opts = faiss.get_compile_options().split()
print("Compile options:", opts)
assert "METAL" in opts, f"METAL not in compile options: {opts}"
res = faiss.StandardMetalResources()
index = faiss.MetalIndexFlatL2(res, 4)
x = np.arange(8, dtype="float32").reshape(2, 4)
index.add(x)
D, I = index.search(x, 1)
assert I.shape == (2, 1)
print("Metal smoke test passed")
PY

uv pip install --python "${VENV_DIR}/bin/python" pytest torch
"${VENV_DIR}/bin/python" -X faulthandler - <<'PY'
import faiss
import torch
import faiss.contrib.torch_utils

index = faiss.IndexFlatL2(32)
xb = torch.rand(256, 32, dtype=torch.float32)
xq = torch.rand(8, 32, dtype=torch.float32)
index.add(xb)
labels = index.assign(xq, 5)
assert labels.shape == (8, 5)
print("Faiss-first torch interop smoke test passed")
PY
"${VENV_DIR}/bin/python" -X faulthandler - <<'PY'
import torch
import faiss
import faiss.contrib.torch_utils

index = faiss.IndexFlatL2(32)
xb = torch.rand(256, 32, dtype=torch.float32)
xq = torch.rand(8, 32, dtype=torch.float32)
index.add(xb)
labels = index.assign(xq, 5)
assert labels.shape == (8, 5)
print("Torch-first faiss interop smoke test passed")
PY
"${VENV_DIR}/bin/python" -m pytest --junitxml=test-results/pytest/results.xml tests/test_*.py
"${VENV_DIR}/bin/python" -m pytest --junitxml=test-results/pytest/results-torch.xml tests/torch_*.py
"${VENV_DIR}/bin/python" -m pytest --junitxml=test-results/pytest/results-metal.xml faiss/metal/test/test_metal_*.py
