#!/bin/bash -l
# GPU benchmark on one NVIDIA GPU: tests, then PyTorch on CUDA and the Triton kernels in float32 and float64 from 473 to
# 112,331 atoms and up to ~450,000 atoms, plus the Rust kernels on the job's CPU cores (also the float64 accuracy
# reference). Defaults to one A100 (a100el9: 32 cores per GPU). For the other GPU partitions, override at submission:
#   sbatch hpc/niflheim/gpu_benchmark.sbatch                                              # A100 (setup.sh on hrid)
#   sbatch --partition=h200el9   --cpus-per-task=24 hpc/niflheim/gpu_benchmark.sbatch     # H200 (setup.sh on sara)
#   sbatch --partition=sm3090el9 --cpus-per-task=8  hpc/niflheim/gpu_benchmark.sbatch     # RTX 3090 (setup.sh on vid)
# Submit from the repository root. The GPU partitions need group authorization.
#SBATCH --job-name=dc-gpu-bench
#SBATCH --partition=a100el9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --output=hpc_results/slurm-%x-%j.out

REPO=${SLURM_SUBMIT_DIR:-$(pwd)}
source "$REPO/hpc/niflheim/job_common.sh"

dc_run_tests

echo "== $(date) benchmark"
python kernels/benchmark.py \
    --backends rust cuda triton \
    --dtypes float32 float64 \
    --radii ${DC_RADII:-10 20 30 45 63} \
    --extra-radii ${DC_EXTRA_RADII:-80 100} --extra-backends rust cuda triton \
    --label "$PARTITION" \
    --out "$OUT/results.jsonl" --arrays-dir "$JOB_TMP/arrays" 2>&1 | tee "$OUT/benchmark.log"

echo "== $(date) done; results in $OUT"
