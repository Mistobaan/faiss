#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/install-metal-deps-osx-arm64.sh"
"${SCRIPT_DIR}/build-metal-wheel-osx-arm64.sh"
"${SCRIPT_DIR}/test-metal-wheel-osx-arm64.sh"
