#!/bin/bash
# One-time setup of DebyeCalculator with the Rust, C and Triton kernels on Niflheim.
#
# Uses the Niflheim modules for Python and Rust; PyTorch (with CUDA and Triton) comes from pip, since Niflheim has
# no PyTorch module. Run it on the login node that matches the partitions you will use (it builds for that node's
# CPU architecture):
#   hrid  -> xeon56el9 (CPU benchmark) and a100el9
#   sara  -> h200el9
#   vid   -> sm3090el9
# from the repository root:
#   bash hpc/niflheim/setup.sh
# Re-running it updates the environment. Needs internet access (pip, crates.io, rustup), which login nodes have.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
source "$REPO/hpc/niflheim/env.sh"

ARCH=${1:-${CPU_ARCH:-}}
if [ -z "$ARCH" ]; then
    echo "CPU_ARCH is not set; give the architecture as an argument, e.g.: bash hpc/niflheim/setup.sh icelake_el9" >&2
    exit 1
fi
VENV=$(dc_venv "$ARCH")
echo "== Setting up $VENV for CPU_ARCH=$ARCH on $(hostname)"

dc_load_modules
python --version

if [ ! -x "$VENV/bin/python" ]; then
    mkdir -p "$DC_VENV_ROOT"
    python -m venv "$VENV"
fi
source "$VENV/bin/activate"
python -m pip install --upgrade pip setuptools wheel

echo "== PyTorch with CUDA from $DC_TORCH_INDEX (includes Triton on Linux)"
python -m pip install torch --index-url "$DC_TORCH_INDEX"

echo "== DebyeCalculator (editable) and benchmark tools"
python -m pip install -e "$REPO" pytest psutil maturin

echo "== Rust toolchain (module $DC_RUST_MODULE, else rustup)"
if ! command -v cargo >/dev/null 2>&1; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o "${TMPDIR:-/tmp}/rustup-init.sh"
    sh "${TMPDIR:-/tmp}/rustup-init.sh" -y --profile minimal --no-modify-path
    rm -f "${TMPDIR:-/tmp}/rustup-init.sh"
    export PATH="$HOME/.cargo/bin:$PATH"
fi
cargo --version

echo "== Rust and C kernels (portable builds; the batch jobs rebuild them for the compute node's CPU)"
python -m pip install "$REPO/kernels" "$REPO/kernels_c"
# Download the Rust dependencies now, so the batch jobs can build without network access
cargo fetch --manifest-path "$REPO/kernels/Cargo.toml"

echo "== Check"
python - <<'EOF'
import torch
from debyecalculator.debye_calculator import _NATIVE_BACKENDS
print('torch', torch.__version__, '| CUDA build', torch.version.cuda)
print('native CPU kernels:', sorted(_NATIVE_BACKENDS))
try:
    import triton
    print('triton', triton.__version__)
except ImportError:
    print('triton: not installed (the GPU jobs will only run the PyTorch CUDA code)')
EOF
echo "== Done: $VENV"
