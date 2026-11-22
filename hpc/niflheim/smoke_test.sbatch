#!/bin/bash -l
# Quick check (about 10 minutes) before the long benchmarks: the test suite on one GPU, and a small benchmark of every
# backend (473 and 3,833 atoms). Same partition options as gpu_benchmark.sbatch:
#   sbatch hpc/niflheim/smoke_test.sbatch
#   sbatch --partition=h200el9   --cpus-per-task=24 hpc/niflheim/smoke_test.sbatch
#   sbatch --partition=sm3090el9 --cpus-per-task=8  hpc/niflheim/smoke_test.sbatch
#SBATCH --job-name=dc-smoke
#SBATCH --partition=a100el9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=hpc_results/slurm-%x-%j.out

REPO=${SLURM_SUBMIT_DIR:-$(pwd)}
source "$REPO/hpc/niflheim/job_common.sh"

dc_run_tests

echo "== $(date) small benchmark"
python kernels/benchmark.py --backends torch rust c cuda triton --dtypes float32 float64 --radii 10 20 \
    --label "${PARTITION}_smoke" --out "$OUT/results.jsonl" --arrays-dir "$JOB_TMP/arrays" 2>&1 | tee "$OUT/benchmark.log"

echo "== $(date) done; results in $OUT"
