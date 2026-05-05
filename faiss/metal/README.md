# Faiss Metal Backend

This directory contains the Apple Silicon Metal backend for Faiss.

The backend is intentionally narrower than the CUDA/HIP code paths. In its
current form it targets:

- macOS on Apple Silicon
- device `0` only
- flat exhaustive search via `MetalIndexFlatL2` and `MetalIndexFlatIP`
- TurboQuant via `MetalIndexTurboQuantMSE`

It is not a generic "GPU=True on macOS" switch for the rest of Faiss. The
backend is a separate implementation layer under `faiss::metal`, with its own
build target, packaging rules, runtime probe, and Python loader.

## Overview

The public C++ API mirrors the shape of the existing device-backed Faiss APIs
where that helps interoperability:

- `StandardMetalResources`
- `MetalIndexFlat`
- `MetalIndexFlatL2`
- `MetalIndexFlatIP`
- `MetalIndexTurboQuantMSE`
- `index_cpu_to_metal`
- `index_metal_to_cpu`

Current limitations:

- Apple Silicon only
- `device = 0` only
- no generic parity with the full `faiss/gpu` surface
- no separate Python ABI-specific Metal extension module

## Architecture

The backend is split into a few clear layers.

### C++ API

The public C++ entry points live in `namespace faiss::metal`.

- `StandardMetalResources` is the resource/provider object used by Metal
  indexes.
- `MetalIndex` is the common base for Metal-backed Faiss indexes.
- `MetalIndexFlat*` implements flat exhaustive search.
- `MetalIndexTurboQuantMSE` preserves the existing TurboQuant code layout while
  providing Metal-backed add/search/reconstruct behavior.
- `index_cpu_to_metal` and `index_metal_to_cpu` provide cloning between CPU and
  Metal indexes.

### Backend Implementation

The implementation details live under `impl/`.

- `impl/MetalResourcesImpl.mm` manages the Metal device, command queue,
  embedded shader library, and pipeline setup.
- `impl/MetalFlatIndex.*` stores vector data in `MTLBuffer`s and runs flat
  search kernels.
- `impl/MetalTurboQuantizer.*` handles Metal-side TurboQuant encode/decode
  helpers.
- `impl/FlatSearchKernels.metal` contains the flat-search kernels.
- `impl/TurboQuantKernels.metal` contains the TurboQuant kernels.

### C Bridge

`MetalCAPI.cpp` exposes a small C ABI over the C++ backend. This is the stable
boundary used by the Python additive loader.

The main exported families are:

- `faiss_metal_runtime_available`
- `faiss_metal_resources_*`
- `faiss_metal_index_flat_new`
- `faiss_metal_index_turbo_quant_new`
- `faiss_metal_index_from_cpu`
- `faiss_metal_index_to_cpu`
- `faiss_metal_index_add/search/reconstruct/...`

### Python Loader

The Python-side additive loader lives in `../python/_metal.py`.

It loads `libfaiss.dylib` with `ctypes`, probes runtime availability, and, if
successful, installs the Metal symbols into the top-level `faiss` namespace.

That means users still write:

```python
import faiss

res = faiss.StandardMetalResources()
index = faiss.MetalIndexFlatL2(res, 128)
```

## Entry Points

### C++

These are the primary entry points when using the backend from C++:

- `StandardMetalResources`
- `MetalIndexFlatL2`
- `MetalIndexFlatIP`
- `MetalIndexTurboQuantMSE`
- `index_cpu_to_metal`
- `index_metal_to_cpu`

`MetalIndexFlat*` supports direct construction from dimensions or copy
construction from CPU `IndexFlat*` instances. `MetalIndexTurboQuantMSE` supports
direct construction from `(d, nbits, metric, seed, store_norm)` or copy
construction from a CPU `IndexTurboQuantMSE`.

### C Bridge

These functions matter if the backend is being loaded dynamically or from a
non-C++ host:

- `faiss_metal_runtime_available`
- `faiss_metal_resources_new` / `free`
- `faiss_metal_resources_get_device`
- `faiss_metal_resources_is_apple_silicon`
- `faiss_metal_index_flat_new`
- `faiss_metal_index_turbo_quant_new`
- `faiss_metal_index_from_cpu`
- `faiss_metal_index_to_cpu`

The bridge returns opaque handles and uses `faiss_metal_last_error()` for error
reporting.

### Python

When the Metal-enabled library is present and the runtime probe succeeds, the
top-level `faiss` module exposes:

- `faiss.StandardMetalResources`
- `faiss.MetalIndexFlatL2`
- `faiss.MetalIndexFlatIP`
- `faiss.MetalIndexTurboQuantMSE`
- `faiss.index_cpu_to_metal`
- `faiss.index_metal_to_cpu`

`faiss.get_compile_options()` appends `METAL` only when the backend is both
present and usable at runtime.

## Packaging Decisions

On Apple Silicon, Metal is part of the main `faiss` package.

### Base Wheel / Package: `faiss`

The base wheel keeps the existing SWIG-backed Python extension:

- `faiss/_swigfaiss.<PEP 3149 suffix>`
- `faiss/_metal.py`
- `faiss/lib/libfaiss.dylib`
- `faiss/lib/libomp.dylib` (vendored fallback for the repaired macOS wheel)

This wheel is CPython ABI-specific because `_swigfaiss` is a real CPython
extension module that links against the Python C API.

On `osx-arm64`, the base package includes the Metal backend by default. There
is no separate `faiss-metal` artifact in the current packaging model.

To avoid loading two different LLVM OpenMP runtimes in one process, the Metal
wheel is repaired after build time so both `faiss/_swigfaiss` and
`faiss/lib/libfaiss.dylib` depend on `@rpath/libomp.dylib` instead of a
Homebrew absolute path. The repaired wheel searches `torch/lib` first so it
reuses PyTorch's bundled runtime when available, then falls back to the
vendored `faiss/lib/libomp.dylib` for standalone installs.

### Platform Tags

The repository does not force a specific macOS wheel tag such as
`macosx_14_0_arm64`.

Instead, the effective wheel platform tag comes from the build environment and
the interpreter configuration, including `MACOSX_DEPLOYMENT_TARGET` when set.

### Shader Packaging

The preferred release path is:

1. combine the Metal family files into one generated source
2. link the result into a single `metallib`
3. embed that `metallib` into `libfaiss.dylib`

This follows the same broad idea used by PyTorch’s MPS packaging: ship your own
backend artifact, not loose source files or Apple frameworks.

If the offline Metal compiler is unavailable, the current build falls back to
embedding Metal source and compiling it at runtime. That fallback is useful for
development, but release artifacts should prefer the offline-compiled path.

## PEPs and Tagging

The relevant packaging conventions are:

- **PEP 517**: `pyproject.toml` declares a standards-based Python build system.
  This repo currently uses `setuptools.build_meta`.
- **PEP 3149**: the base CPython extension uses the normal ABI-tagged filename
  convention, e.g. `_swigfaiss.cpython-314-darwin.so`.
- **PEP 425** and **PEP 427**: wheel compatibility tags and wheel filename
  format.

The practical outcome in this backend is a single CPython ABI-specific `faiss`
wheel on Apple Silicon.

## Build and Test

Run these commands from the repository root.

### Requirements

- macOS on Apple Silicon
- Xcode command line tools
- `xcrun`, `metal`, and `metallib` for the offline shader path
- CMake 3.24+
- BLAS/OpenMP dependencies required by Faiss
- Python environment with `swig` when building the base Python extension

### Native Build

```bash
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
PREFIX="${CONDA_PREFIX:-$(cd "$(dirname "$PYTHON_BIN")/.." && pwd)}"
OPENMP_PREFIX="${OpenMP_ROOT:-$PREFIX}"
JOBS="${CMAKE_BUILD_PARALLEL_LEVEL:-$(sysctl -n hw.ncpu 2>/dev/null || echo 8)}"

cmake -S . -B build \
  -DBUILD_SHARED_LIBS=ON \
  -DFAISS_ENABLE_GPU=OFF \
  -DFAISS_ENABLE_METAL=ON \
  -DFAISS_ENABLE_PYTHON=ON \
  -DBUILD_TESTING=ON \
  -DFETCHCONTENT_UPDATES_DISCONNECTED=ON \
  -DPython_EXECUTABLE="$PYTHON_BIN" \
  -DCMAKE_PREFIX_PATH="$PREFIX" \
  -DOpenMP_ROOT="$OPENMP_PREFIX"

cmake --build build -j "$JOBS" --target faiss swigfaiss faiss_test faiss_metal_test
```

### C++ Metal Tests

```bash
ctest --test-dir build --output-on-failure -R TestMetal
```

### CI-Style Wheel Build And Test

Use the same repo script that the macOS Metal workflow calls:

```bash
./scripts/ci/run-metal-wheel-osx-arm64.sh
```

That flow installs the Homebrew dependencies, builds the Metal wheel into
`dist/`, repairs the wheel's OpenMP load commands, creates `.venv-test`, runs
the installed-wheel smoke test, and then executes:

- `tests/test_*.py`
- `tests/torch_*.py`
- `faiss/metal/test/test_metal_*.py`

You can also run the individual steps directly:

- `./scripts/ci/install-metal-deps-osx-arm64.sh`
- `./scripts/ci/build-metal-wheel-osx-arm64.sh`
- `./scripts/ci/test-metal-wheel-osx-arm64.sh`

`DIST_DIR` and `VENV_DIR` may be overridden when you want alternate output or
virtualenv locations.

## Implementation Notes

- The repo-tracked kernels live in `impl/FlatSearchKernels.metal` and
  `impl/TurboQuantKernels.metal`.
- The build generates a self-contained combined Metal source before either the
  offline compile path or the runtime-source fallback path.
- `MetalIndexFlat*` stores vectors in Metal buffers and runs exhaustive flat
  distance kernels plus top-k selection.
- `MetalIndexTurboQuantMSE` preserves the CPU `TurboQuantizer` metadata and code
  layout, then keeps Metal-side decoded storage for flat search over the
  reconstructed vectors.
- Resource initialization happens eagerly enough to make runtime probing work
  before the first explicit index construction.
- Python backend discovery is additive:
  - the base package imports normally
  - `faiss._metal` looks for `libfaiss.dylib`
  - if the runtime probe succeeds, the Metal API is installed into `faiss`

## Debugging Tips

- Rebuild after changes to either Metal family file.
- If shader loading fails, start by checking whether the build used offline
  `metal` / `metallib` compilation or the runtime-compilation fallback.
- If the Python API does not expose Metal symbols, check both:
  - whether `libfaiss.dylib` is discoverable from the active environment
  - whether `faiss_metal_runtime_available()` succeeds on that machine
