# Shared environment for the Niflheim scripts (sourced by setup.sh and the sbatch scripts).
#
# One Python virtual environment per CPU architecture, because Niflheim's node types (and their modules) differ:
#   $DC_VENV_ROOT/debyecalculator-<arch>       (default DC_VENV_ROOT=$HOME/venvs)
# Settings (export before running a script to change them):
#   DC_PYTHON_MODULE  Python module               (default Python/3.13.5-GCCcore-14.3.0)
#   DC_VENV_ROOT      where the venvs live        (default $HOME/venvs)
#   DC_TORCH_INDEX    pip index of the PyTorch CUDA wheels (default https://download.pytorch.org/whl/cu126)

DC_PYTHON_MODULE=${DC_PYTHON_MODULE:-Python/3.13.5-GCCcore-14.3.0}
DC_VENV_ROOT=${DC_VENV_ROOT:-$HOME/venvs}
DC_TORCH_INDEX=${DC_TORCH_INDEX:-https://download.pytorch.org/whl/cu126}

# CPU architecture of a partition, as the CPU_ARCH of its login node (see the Niflheim "Getting started" page)
dc_arch_for_partition() {
    case "$1" in
        xeon56el9*|a100el9*) echo icelake_el9 ;;      # login node hrid
        xeon32el9*|h200el9*) echo sapphirerapids ;;   # login node sara
        xeon40el9*|sm3090el9*) echo skylake_el9 ;;    # login nodes vid, thul
        epyc96el9*) echo epyc9004_el9 ;;              # login node fjorm
        xeon24el9*) echo broadwell_el9 ;;             # login node slid
        *) echo "${CPU_ARCH:-unknown}" ;;
    esac
}

dc_venv() {
    echo "$DC_VENV_ROOT/debyecalculator-$1"
}

# Rust toolchain installed by rustup in the home directory (shared by all nodes)
if [ -x "$HOME/.cargo/bin/cargo" ]; then
    export PATH="$HOME/.cargo/bin:$PATH"
fi
