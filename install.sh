#!/usr/bin/env bash
# install.sh — build and install the isthmus Python package.
#
# Usage:
#   ./install.sh              # standard install (into the active Python env)
#   ./install.sh --editable   # editable / development install
#   ./install.sh --user       # install to ~/.local (no root required)
#   ./install.sh --prefix /my/path  # custom install prefix

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EDITABLE=0
PIP_EXTRA_ARGS=()

# Parse arguments
for arg in "$@"; do
    case "$arg" in
        --editable|-e)
            EDITABLE=1
            ;;
        --user)
            PIP_EXTRA_ARGS+=("--user")
            ;;
        --prefix)
            shift
            PIP_EXTRA_ARGS+=("--prefix" "$1")
            ;;
        --help|-h)
            echo "Usage: $0 [--editable] [--user] [--prefix PATH]"
            exit 0
            ;;
    esac
done

echo "========================================"
echo "  ISTHMUS Python Wrapper Installer"
echo "========================================"
echo ""

# ---- Prerequisite checks ---------------------------------------------------

check() {
    if ! command -v "$1" &>/dev/null; then
        echo "ERROR: '$1' not found. $2"
        exit 1
    fi
}

check python3  "Install Python 3.8 or newer."
check pip3     "Install pip (python3 -m ensurepip)."
check cmake    "Install CMake 3.22 or newer (https://cmake.org/download/)."
check git      "Install git (needed to fetch pybind11 if not pre-installed)."

PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
CMAKE_VERSION=$(cmake --version | awk 'NR==1{print $3}')
echo "Python  : ${PY_VERSION}"
echo "CMake   : ${CMAKE_VERSION}"
echo "Install : ${SCRIPT_DIR}"
echo ""

# ---- Install Python build dependencies -------------------------------------

echo "Installing build dependencies (pybind11, scikit-build-core, ninja)..."
pip3 install --upgrade "pybind11>=2.11" "scikit-build-core>=0.5" "ninja"
echo ""

# ---- Build and install the isthmus package ---------------------------------

cd "${SCRIPT_DIR}"

if [[ "${EDITABLE}" -eq 1 ]]; then
    echo "Installing in editable (development) mode..."
    pip3 install -e . --no-build-isolation "${PIP_EXTRA_ARGS[@]}"
else
    echo "Building and installing isthmus..."
    pip3 install . "${PIP_EXTRA_ARGS[@]}"
fi

# ---- Smoke test ------------------------------------------------------------

echo ""
echo "Verifying installation..."
python3 - <<'EOF'
import isthmus
mw = isthmus.MarchingWindows()
print(f"  isthmus version : {isthmus.__version__}")
print(f"  MarchingWindows : {mw}")
print("  OK")
EOF

echo ""
echo "========================================"
echo "  Installation complete!"
echo "========================================"
echo ""
echo "Try the quick-start example:"
echo "  python3 python/examples/basic_run.py"
