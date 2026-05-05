#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

DIST_DIR="${DIST_DIR:-dist}"
CONDA_ENV_PREFIX="${CONDA_PREFIX:-}"

if [[ -z "${CONDA_ENV_PREFIX}" ]]; then
  echo "CONDA_PREFIX is not set. Activate the conda environment before repairing the wheel." >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV_PREFIX}/bin/python}"
LIBOMP_PREFIX="${LIBOMP_PREFIX:-${CONDA_ENV_PREFIX}}"
LIBOMP_DYLIB="${LIBOMP_DYLIB:-${LIBOMP_PREFIX}/lib/libomp.dylib}"
OPENBLAS_PREFIX="${OPENBLAS_PREFIX:-${CONDA_ENV_PREFIX}}"
OPENBLAS_RUNTIME_DYLIB="${OPENBLAS_RUNTIME_DYLIB:-${OPENBLAS_PREFIX}/lib/libopenblas.0.dylib}"
GFORTRAN_DYLIB="${GFORTRAN_DYLIB:-${OPENBLAS_PREFIX}/lib/libgfortran.5.dylib}"
QUADMATH_DYLIB="${QUADMATH_DYLIB:-${OPENBLAS_PREFIX}/lib/libquadmath.0.dylib}"
LIBGCC_DYLIB="${LIBGCC_DYLIB:-${OPENBLAS_PREFIX}/lib/libgcc_s.1.1.dylib}"
WHEEL_PATH="${1:-${WHEEL_PATH:-}}"

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

if [ ! -f "${WHEEL_PATH}" ]; then
  echo "Wheel not found: ${WHEEL_PATH}" >&2
  exit 1
fi

if [ ! -f "${LIBOMP_DYLIB}" ]; then
  echo "libomp dylib not found: ${LIBOMP_DYLIB}" >&2
  exit 1
fi

for dylib in "${OPENBLAS_RUNTIME_DYLIB}" "${GFORTRAN_DYLIB}" "${QUADMATH_DYLIB}" "${LIBGCC_DYLIB}"; do
  if [ ! -f "${dylib}" ]; then
    echo "Runtime dylib not found: ${dylib}" >&2
    exit 1
  fi
done

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "Python not found in active conda environment: ${PYTHON_BIN}" >&2
  exit 1
fi

TMP_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

UNPACK_DIR="${TMP_DIR}/wheel"
mkdir -p "${UNPACK_DIR}"

"${PYTHON_BIN}" - "${WHEEL_PATH}" "${UNPACK_DIR}" <<'PY'
import sys
import zipfile

wheel_path, unpack_dir = sys.argv[1:3]
with zipfile.ZipFile(wheel_path) as wheel:
    wheel.extractall(unpack_dir)
PY

FAISS_PKG_DIR="${UNPACK_DIR}/faiss"
FAISS_LIB_DIR="${FAISS_PKG_DIR}/lib"
SWIG_SO="$(find "${FAISS_PKG_DIR}" -maxdepth 1 -type f -name '_swigfaiss*.so' | head -n 1)"
LIBFAISS_DYLIB="${FAISS_LIB_DIR}/libfaiss.dylib"
VENDORED_LIBOMP="${FAISS_LIB_DIR}/libomp.dylib"
VENDORED_OPENBLAS="${FAISS_LIB_DIR}/$(basename "${OPENBLAS_RUNTIME_DYLIB}")"
VENDORED_GFORTRAN="${FAISS_LIB_DIR}/$(basename "${GFORTRAN_DYLIB}")"
VENDORED_QUADMATH="${FAISS_LIB_DIR}/$(basename "${QUADMATH_DYLIB}")"
VENDORED_LIBGCC="${FAISS_LIB_DIR}/$(basename "${LIBGCC_DYLIB}")"

if [ -z "${SWIG_SO}" ] || [ ! -f "${LIBFAISS_DYLIB}" ]; then
  echo "Expected faiss wheel layout not found under ${UNPACK_DIR}" >&2
  exit 1
fi

vendor_dylib() {
  local source="$1"
  local destination="$2"

  cp "${source}" "${destination}"
  chmod 755 "${destination}"
  install_name_tool -id "@rpath/$(basename "${destination}")" "${destination}"
}

vendor_dylib "${LIBOMP_DYLIB}" "${VENDORED_LIBOMP}"
vendor_dylib "${OPENBLAS_RUNTIME_DYLIB}" "${VENDORED_OPENBLAS}"
vendor_dylib "${GFORTRAN_DYLIB}" "${VENDORED_GFORTRAN}"
vendor_dylib "${QUADMATH_DYLIB}" "${VENDORED_QUADMATH}"
vendor_dylib "${LIBGCC_DYLIB}" "${VENDORED_LIBGCC}"

list_rpaths() {
  otool -l "$1" | awk '/cmd LC_RPATH/ { getline; getline; print $2 }'
}

delete_rpath_if_present() {
  local binary="$1"
  local rpath="$2"

  if list_rpaths "${binary}" | grep -Fxq "${rpath}"; then
    install_name_tool -delete_rpath "${rpath}" "${binary}"
  fi
}

add_rpath_if_missing() {
  local binary="$1"
  local rpath="$2"

  if ! list_rpaths "${binary}" | grep -Fxq "${rpath}"; then
    install_name_tool -add_rpath "${rpath}" "${binary}"
  fi
}

rewrite_dependency() {
  local binary="$1"
  local old_path="$2"
  local new_path="$3"

  if otool -L "${binary}" | grep -Fq "${old_path}"; then
    install_name_tool -change "${old_path}" "${new_path}" "${binary}"
  fi
}

rewrite_dependency "${SWIG_SO}" "${LIBOMP_DYLIB}" "@rpath/libomp.dylib"
rewrite_dependency "${LIBFAISS_DYLIB}" "${LIBOMP_DYLIB}" "@rpath/libomp.dylib"
rewrite_dependency "${LIBFAISS_DYLIB}" "${OPENBLAS_RUNTIME_DYLIB}" "@rpath/$(basename "${OPENBLAS_RUNTIME_DYLIB}")"
rewrite_dependency "${VENDORED_OPENBLAS}" "${GFORTRAN_DYLIB}" "@rpath/$(basename "${GFORTRAN_DYLIB}")"
rewrite_dependency "${VENDORED_OPENBLAS}" "${LIBOMP_DYLIB}" "@rpath/libomp.dylib"
rewrite_dependency "${VENDORED_GFORTRAN}" "${QUADMATH_DYLIB}" "@rpath/$(basename "${QUADMATH_DYLIB}")"
rewrite_dependency "${VENDORED_GFORTRAN}" "${LIBGCC_DYLIB}" "@rpath/$(basename "${LIBGCC_DYLIB}")"

delete_rpath_if_present "${SWIG_SO}" "@loader_path/../torch/lib"
delete_rpath_if_present "${SWIG_SO}" "@loader_path/lib"
add_rpath_if_missing "${SWIG_SO}" "@loader_path/../torch/lib"
add_rpath_if_missing "${SWIG_SO}" "@loader_path/lib"

delete_rpath_if_present "${LIBFAISS_DYLIB}" "@loader_path/../../torch/lib"
delete_rpath_if_present "${LIBFAISS_DYLIB}" "@loader_path"
add_rpath_if_missing "${LIBFAISS_DYLIB}" "@loader_path/../../torch/lib"
add_rpath_if_missing "${LIBFAISS_DYLIB}" "@loader_path"

if command -v codesign >/dev/null 2>&1; then
  codesign --force --sign - --timestamp=none "${VENDORED_LIBOMP}" >/dev/null
  codesign --force --sign - --timestamp=none "${VENDORED_OPENBLAS}" >/dev/null
  codesign --force --sign - --timestamp=none "${VENDORED_GFORTRAN}" >/dev/null
  codesign --force --sign - --timestamp=none "${VENDORED_QUADMATH}" >/dev/null
  codesign --force --sign - --timestamp=none "${VENDORED_LIBGCC}" >/dev/null
  codesign --force --sign - --timestamp=none "${LIBFAISS_DYLIB}" >/dev/null
  codesign --force --sign - --timestamp=none "${SWIG_SO}" >/dev/null
fi

"${PYTHON_BIN}" - "${UNPACK_DIR}" "${WHEEL_PATH}" <<'PY'
import base64
import csv
import hashlib
import pathlib
import sys
import zipfile

root = pathlib.Path(sys.argv[1])
wheel_path = pathlib.Path(sys.argv[2])
tmp_wheel_path = wheel_path.with_suffix(wheel_path.suffix + ".tmp")

dist_info_dirs = sorted(root.glob("*.dist-info"))
if len(dist_info_dirs) != 1:
    raise SystemExit(f"Expected exactly one .dist-info directory in {root}")

record_path = dist_info_dirs[0] / "RECORD"
record_relpath = record_path.relative_to(root).as_posix()

rows = []
files = sorted(path for path in root.rglob("*") if path.is_file())
for path in files:
    relpath = path.relative_to(root).as_posix()
    if relpath == record_relpath:
        continue

    data = path.read_bytes()
    digest = hashlib.sha256(data).digest()
    digest_b64 = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    rows.append((relpath, f"sha256={digest_b64}", str(len(data))))

rows.append((record_relpath, "", ""))
with record_path.open("w", newline="") as record_file:
    writer = csv.writer(record_file)
    writer.writerows(rows)

with zipfile.ZipFile(tmp_wheel_path, "w", compression=zipfile.ZIP_DEFLATED) as wheel:
    for path in sorted(path for path in root.rglob("*") if path.is_file()):
        relpath = path.relative_to(root).as_posix()
        info = zipfile.ZipInfo.from_file(path, relpath)
        info.compress_type = zipfile.ZIP_DEFLATED
        with path.open("rb") as source:
            wheel.writestr(info, source.read())

tmp_wheel_path.replace(wheel_path)
PY

echo "Repaired macOS wheel conda runtime linkage: ${WHEEL_PATH}"
