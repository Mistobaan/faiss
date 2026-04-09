# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Legacy setup.py — kept for the classic CMake workflow:

    cmake -S . -B build ...
    cmake --build build --target faiss swigfaiss
    cd build/faiss/python && python setup.py install

For modern installs prefer:

    pip install faiss/python/
    uv pip install faiss/python/

which are handled entirely by pyproject.toml + scikit-build-core.
"""
from __future__ import print_function

import os
import platform
import shutil
import sysconfig

from setuptools import setup

# ── assemble the faiss Python package from pre-built artefacts ────────────
shutil.rmtree("faiss", ignore_errors=True)
os.mkdir("faiss")
shutil.copytree("contrib", "faiss/contrib")
shutil.copyfile("__init__.py", "faiss/__init__.py")
shutil.copyfile("loader.py", "faiss/loader.py")
shutil.copyfile("class_wrappers.py", "faiss/class_wrappers.py")
shutil.copyfile("gpu_wrappers.py", "faiss/gpu_wrappers.py")
shutil.copyfile("extra_wrappers.py", "faiss/extra_wrappers.py")
shutil.copyfile("array_conversions.py", "faiss/array_conversions.py")

if os.path.exists("__init__.pyi"):
    shutil.copyfile("__init__.pyi", "faiss/__init__.pyi")
if os.path.exists("py.typed"):
    shutil.copyfile("py.typed", "faiss/py.typed")

# ── locate pre-built SWIG extensions ──────────────────────────────────────
# Use the PEP 3149 compliant extension suffix so the shared object filename
# matches what Python's import machinery expects, e.g.:
#   _swigfaiss.cpython-312-darwin.so   (macOS)
#   _swigfaiss.cpython-312-x86_64-linux-gnu.so   (Linux)
if platform.system() == "AIX":
    ext = ".a"
else:
    # sysconfig.get_config_var('EXT_SUFFIX') returns the full PEP 3149 suffix
    # including the leading dot, e.g. ".cpython-312-darwin.so" or ".pyd".
    ext = sysconfig.get_config_var("EXT_SUFFIX") or (
        ".pyd" if platform.system() == "Windows" else ".so"
    )
prefix = "Release/" * (platform.system() == "Windows")

_VARIANTS = {
    "swigfaiss":             f"{prefix}_swigfaiss{ext}",
    "swigfaiss_avx2":        f"{prefix}_swigfaiss_avx2{ext}",
    "swigfaiss_avx512":      f"{prefix}_swigfaiss_avx512{ext}",
    "swigfaiss_avx512_spr":  f"{prefix}_swigfaiss_avx512_spr{ext}",
    "swigfaiss_sve":         f"{prefix}_swigfaiss_sve{ext}",
}

found_any = False
for mod_name, lib_path in _VARIANTS.items():
    if os.path.exists(lib_path):
        print(f"Copying {lib_path}")
        shutil.copyfile(f"{mod_name}.py", f"faiss/{mod_name}.py")
        shutil.copyfile(lib_path, f"faiss/_{mod_name}{ext}")
        found_any = True

# callbacks helper library — this is a plain shared lib, not a Python extension,
# so it keeps the simple .so / .pyd suffix.
_plain_ext = ".pyd" if platform.system() == "Windows" else ".so"
callbacks_lib = f"{prefix}libfaiss_python_callbacks{_plain_ext}"
if os.path.exists(callbacks_lib):
    print(f"Copying {callbacks_lib}")
    shutil.copyfile(callbacks_lib, f"faiss/{callbacks_lib}")

# example external module
example_lib = f"_faiss_example_external_module{ext}"
if os.path.exists(example_lib):
    print(f"Copying {example_lib}")
    shutil.copyfile("faiss_example_external_module.py",
                     "faiss/faiss_example_external_module.py")
    shutil.copyfile(example_lib, f"faiss/{example_lib}")
    found_any = True

if platform.system() != "AIX":
    assert found_any, (
        f"Could not find any SWIG extension ({', '.join(_VARIANTS.values())}). "
        f"Faiss may not be compiled yet."
    )

# ── setuptools metadata ──────────────────────────────────────────────────
setup(
    name="faiss",
    version="1.14.1",
    description="A library for efficient similarity search and clustering of dense vectors",
    long_description=(
        "Faiss is a library for efficient similarity search and clustering of dense "
        "vectors. It contains algorithms that search in sets of vectors of any size, "
        "up to ones that possibly do not fit in RAM. It also contains supporting "
        "code for evaluation and parameter tuning. Faiss is written in C++ with "
        "complete wrappers for Python/numpy. Some of the most useful algorithms "
        "are implemented on the GPU. It is developed by Meta AI Research."
    ),
    long_description_content_type="text/plain",
    url="https://github.com/facebookresearch/faiss",
    author="Matthijs Douze, Jeff Johnson, Herve Jegou, Lucas Hosseini",
    author_email="faiss@meta.com",
    license="MIT",
    keywords="search nearest neighbors",
    install_requires=["numpy", "packaging"],
    packages=["faiss", "faiss.contrib", "faiss.contrib.torch"],
    package_data={
        "faiss": ["*.so", "*.pyd", "*.a", "*.pyi", "py.typed"],
    },
    zip_safe=False,
)
