#!/bin/bash
# Submit the CPU benchmark and one GPU benchmark per GPU partition given (default: a100el9), from the repository root:
#   bash hpc/niflheim/submit_all.sh                          # xeon56el9 + a100el9
#   bash hpc/niflheim/submit_all.sh a100el9 h200el9 sm3090el9
# Each partition needs its environment from setup.sh (on hrid for xeon56el9/a100el9, sara for h200el9, vid for
# sm3090el9). DC_SKIP_CPU=1 submits only the GPU jobs.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
source "$REPO/hpc/niflheim/env.sh"
cd "$REPO"
mkdir -p hpc_results

check_env() {
    local venv
    venv=$(dc_venv "$(dc_arch_for_partition "$1")")
    if [ ! -x "$venv/bin/python" ]; then
        echo "skipping $1: no environment $venv (run setup.sh on its login node)" >&2
        return 1
    fi
}

if [ "${DC_SKIP_CPU:-0}" != 1 ] && check_env xeon56el9; then
    sbatch hpc/niflheim/cpu_benchmark.sbatch
fi

for partition in "${@:-a100el9}"; do
    case "$partition" in
        a100el9*) cpus=32 ;;
        h200el9*) cpus=24 ;;
        sm3090el9*) cpus=8 ;;
        *) echo "unknown GPU partition $partition" >&2; continue ;;
    esac
    check_env "$partition" && sbatch --partition="$partition" --cpus-per-task="$cpus" hpc/niflheim/gpu_benchmark.sbatch
done
squeue -u "$USER"
