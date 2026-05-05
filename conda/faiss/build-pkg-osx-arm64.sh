#!/bin/sh
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

set -e


export FAISS_CONDA_SPLIT_INSTALL=1
export FAISS_CONDA_STAGED_PREFIX=_libfaiss_stage
export FAISS_PYTHON_ENABLE_METAL=1

"$PYTHON" -m pip install . --no-deps --no-build-isolation --ignore-installed
