#!/bin/bash -l
# CPU benchmark on one whole xeon56el9 node (Ice Lake, 56 cores): tests, then PyTorch and Rust in float32 and
# float64 from 473 to 112,331 atoms, and Rust up to ~450,000 atoms.
# Submit from the repository root (after setup.sh on hrid):   sbatch hpc/niflheim/cpu_benchmark.sbatch
#SBATCH --job-name=dc-cpu-bench
#SBATCH --partition=xeon56el9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --time=08:00:00
#SBATCH --output=hpc_results/slurm-%x-%j.out

REPO=${SLURM_SUBMIT_DIR:-$(pwd)}
source "$REPO/hpc/niflheim/job_common.sh"

dc_run_tests

echo "== $(date) benchmark"
python kernels/benchmark.py \
    --backends torch rust \
    --dtypes float32 float64 \
    --radii ${DC_RADII:-10 20 30 45 63} \
    --extra-radii ${DC_EXTRA_RADII:-80 100} --extra-backends rust \
    --label "$PARTITION" \
    --out "$OUT/results.jsonl" --arrays-dir "$JOB_TMP/arrays" 2>&1 | tee "$OUT/benchmark.log"

echo "== $(date) done; results in $OUT"
