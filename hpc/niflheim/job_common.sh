# Sourced at the start of the sbatch scripts: environment, results directory, node-specific kernel build,
# machine description and tests. Expects REPO to be set.

set -uo pipefail
source "$REPO/hpc/niflheim/env.sh"

PARTITION=${SLURM_JOB_PARTITION:-local}
ARCH=$(dc_arch_for_partition "$PARTITION")
VENV=$(dc_venv "$ARCH")
OUT="$REPO/hpc_results/${PARTITION}_${SLURM_JOB_ID:-$$}"
mkdir -p "$OUT"
JOB_TMP="${TMPDIR:-/tmp}/debyecalculator_${SLURM_JOB_ID:-$$}"
mkdir -p "$JOB_TMP"
trap 'rm -rf "$JOB_TMP"' EXIT

echo "== $(date) job ${SLURM_JOB_ID:-} on $(hostname), partition $PARTITION, arch $ARCH, results in $OUT"

module purge
module load "$DC_PYTHON_MODULE"
if [ ! -x "$VENV/bin/python" ]; then
    echo "No environment $VENV: run 'bash hpc/niflheim/setup.sh' on the login node for $PARTITION first." >&2
    exit 1
fi
source "$VENV/bin/activate"

# One PyTorch/OpenMP thread per allocated core
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-$(nproc)}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export QT_QPA_PLATFORM=offscreen

# Rebuild the Rust and C kernels for this node's CPU (e.g. AVX-512), offline, into the job's local disk.
# They take precedence over the portable builds in the venv. DC_NATIVE_BUILD=0 keeps the portable builds.
if [ "${DC_NATIVE_BUILD:-1}" = 1 ] && command -v cargo >/dev/null 2>&1; then
    if CARGO_NET_OFFLINE=true CARGO_TARGET_DIR="$JOB_TMP/cargo-target" RUSTFLAGS="-C target-cpu=native" \
        CFLAGS="-march=native" python -m pip install --no-deps --no-build-isolation --no-index \
        --target "$JOB_TMP/native" "$REPO/kernels" "$REPO/kernels_c" > "$OUT/native_build.log" 2>&1; then
        export PYTHONPATH="$JOB_TMP/native${PYTHONPATH:+:$PYTHONPATH}"
        echo "kernels built for this node's CPU (see native_build.log)"
    else
        echo "node-specific kernel build failed (see native_build.log); using the portable builds"
    fi
fi

# What ran where
{
    echo "date: $(date)"; echo "host: $(hostname)"; echo "partition: $PARTITION"; echo "arch: $ARCH"
    echo "slurm: job ${SLURM_JOB_ID:-} cpus ${SLURM_CPUS_PER_TASK:-} gpus ${SLURM_GPUS_ON_NODE:-${CUDA_VISIBLE_DEVICES:-}}"
    echo "git: $(git -C "$REPO" rev-parse --abbrev-ref HEAD) $(git -C "$REPO" rev-parse --short HEAD)"
    echo; lscpu 2>/dev/null
    echo; command -v nvidia-smi >/dev/null && nvidia-smi
    echo; module list 2>&1
    echo; python -m pip freeze
    echo; python - <<'EOF'
import torch
from debyecalculator.debye_calculator import _NATIVE_BACKENDS
print('torch', torch.__version__, 'threads', torch.get_num_threads(), 'cuda', torch.cuda.is_available())
print('native kernels:', {k: v.__file__ for k, v in _NATIVE_BACKENDS.items()})
if torch.cuda.is_available():
    from debyecalculator import triton_pair_sum
    print('GPU', torch.cuda.get_device_name(0))
    print('triton kernels:', 'available' if triton_pair_sum.library() else f'unavailable: {triton_pair_sum.unavailable_reason()}')
EOF
} > "$OUT/environment.txt" 2>&1
tail -4 "$OUT/environment.txt"

cd "$REPO"

dc_run_tests() {
    echo "== $(date) tests"
    python -m pytest -q -p no:cacheprovider -rs debyecalculator > "$OUT/pytest.log" 2>&1
    local status=$?
    tail -3 "$OUT/pytest.log"
    echo "pytest exit status $status" | tee -a "$OUT/pytest.log"
    return 0  # run the benchmark also if some tests fail; the log tells
}
