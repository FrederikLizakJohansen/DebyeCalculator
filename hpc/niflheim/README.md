# Running the tests and benchmarks on Niflheim (DTU)

This runs the test suite and the pair-sum benchmark of every backend on Niflheim:

| job | partition | hardware | backends |
|---|---|---|---|
| CPU benchmark | `xeon56el9` | 1 whole node, Intel Ice Lake, 56 cores | PyTorch CPU, Rust, C |
| GPU benchmark | `a100el9` (default) | 1 × NVIDIA A100, 32 cores | PyTorch CUDA, Triton, Rust (on the 32 cores) |
| | `h200el9` | 1 × NVIDIA H200, 24 cores | same |
| | `sm3090el9` | 1 × NVIDIA RTX 3090, 8 cores | same |

Each job writes JSON-lines results that `kernels/benchmark_report.py` combines with the Mac results
(`kernels/benchmark_results/apple_m3_max.jsonl`) into one table: CPU, MPS and GPU; PyTorch, Rust, C, Metal and
Triton; float32 and float64; time and peak memory.

Facts about Niflheim here come from https://wiki.fysik.dtu.dk/Niflheim_users/Niflheim_Getting_Started/ (read
November 2026). Check there if something has changed.

## 0. Before you start

- You need a Niflheim account, and **your group must be authorized for the GPU partitions** (`a100el9`, `h200el9`,
  `sm3090el9`). Otherwise the GPU jobs stay pending or are rejected. The CPU job needs no authorization.
- Log in from the DTU network, or connect to the DTU VPN first.
- The branch `fast-pair-kernels` must be on GitHub (pushed from the Mac).

## 1. Log in and clone

Use the login node that matches the partitions you will run on. Each login node has the CPU architecture of its
compute nodes, and `setup.sh` builds for it:

| partitions | login node |
|---|---|
| `xeon56el9` (CPU) and `a100el9` | `hrid.fysik.dtu.dk` |
| `h200el9` | `sara.fysik.dtu.dk` |
| `sm3090el9` | `vid.fysik.dtu.dk` (or `thul`) |

Start with `hrid`, which covers the CPU benchmark and the A100:

```bash
ssh <username>@hrid.fysik.dtu.dk
```

```bash
git clone -b fast-pair-kernels https://github.com/FrederikLizakJohansen/DebyeCalculator.git
```

```bash
cd DebyeCalculator
```

Your home directory is shared by all nodes, so clone once. Run every command below from this `DebyeCalculator`
folder.

## 2. Set up the environment (once per login node, ~10–15 minutes)

```bash
bash hpc/niflheim/setup.sh
```

This:

1. loads `Python/3.13.5-GCCcore-14.3.0` and creates the virtual environment `~/venvs/debyecalculator-<CPU_ARCH>`
   (one per architecture, e.g. `debyecalculator-icelake_el9` on `hrid`);
2. installs PyTorch with CUDA 12.6 from the PyTorch wheel index. On Linux these wheels include Triton, and they
   bring their own CUDA libraries, so no CUDA module is needed;
3. installs DebyeCalculator from this folder (editable) plus `pytest`, `psutil` and `maturin`;
4. loads the `Rust/1.88.0-GCCcore-14.3.0` module (rustup into `~/.cargo` only if that module is missing), and builds
   the Rust and C kernels.

Niflheim has no PyTorch or Triton module, so PyTorch comes from pip (about 8 GB in the venv, mostly the CUDA
libraries). The CUDA and cuDNN modules aren't needed: the wheels bring their own.

It ends by printing the versions and `native CPU kernels: ['c', 'rust']`. If you will also use the H200 or the
RTX 3090, log in to `sara` or `vid` and run the same command there; `~/.cargo` and the clone are shared.

Login nodes limit a process to 20 minutes of CPU time. If the setup is stopped, run it again: it continues where it
left off.

## 3. Quick check on a GPU (~10 minutes)

```bash
mkdir -p hpc_results
```

```bash
sbatch hpc/niflheim/smoke_test.sbatch
```

(For the H200: `sbatch --partition=h200el9 --cpus-per-task=24 hpc/niflheim/smoke_test.sbatch`; for the RTX 3090:
`sbatch --partition=sm3090el9 --cpus-per-task=8 hpc/niflheim/smoke_test.sbatch`.)

Follow it with `squeue -u $USER`. When it's done, look at the job's folder `hpc_results/<partition>_<jobid>/`:

- `pytest.log` ends with the test summary. On a GPU node, the CUDA tests and the Triton tests run there for the first
  time;
- `environment.txt` ends with the GPU name and `triton kernels: available`, or the reason why not;
- `benchmark.log` has a small benchmark of every backend.

If tests fail or Triton is unavailable, send me `hpc_results/` (step 6) before running the long jobs.

## 4. The benchmarks

```bash
bash hpc/niflheim/submit_all.sh a100el9
```

This submits the CPU benchmark (`xeon56el9`, a whole node, up to 8 hours) and the A100 benchmark (up to 10 hours).
Add `h200el9` and/or `sm3090el9` to also benchmark those GPUs, after running `setup.sh` on their login nodes:

```bash
bash hpc/niflheim/submit_all.sh a100el9 h200el9 sm3090el9
```

To submit a single job: `sbatch hpc/niflheim/cpu_benchmark.sbatch`, or `sbatch hpc/niflheim/gpu_benchmark.sbatch`
with the same `--partition`/`--cpus-per-task` options as in step 3.

What runs:

- particles of radius 10, 20, 30, 45 and 63 Å (473 to 112,331 atoms) for every backend, and of 80 and 100 Å
  (~230,000 and ~450,000 atoms) for the fast backends (Rust and C on the CPU node; Triton, PyTorch CUDA and Rust on the
  GPU node);
- float32 and float64; I(Q) and I(Q) with the gradient with respect to all positions; time and peak memory (RAM and
  GPU); accuracy against the float64 result of the Rust kernels;
- each job first runs the test suite, and rebuilds the Rust and C kernels for the node's CPU (e.g. AVX-512).

Expected run times: the CPU job is dominated by PyTorch CPU at 112,331 atoms (minutes per configuration); the GPU
jobs by PyTorch CUDA at the largest sizes. Change the sizes with `DC_RADII` and `DC_EXTRA_RADII`, e.g.
`DC_EXTRA_RADII="" sbatch hpc/niflheim/gpu_benchmark.sbatch` to skip the largest particles.

Monitor:

```bash
squeue -u $USER
```

```bash
tail -f hpc_results/slurm-dc-cpu-bench-<jobid>.out
```

## 5. Results

Each job writes `hpc_results/<partition>_<jobid>/`:

| file | contents |
|---|---|
| `results.jsonl` | one line per backend, precision and size: times, memory, accuracy, machine |
| `benchmark.log` | the same as a readable log |
| `pytest.log` | test results |
| `environment.txt` | node, CPU (`lscpu`), GPU (`nvidia-smi`), modules, Python packages |
| `native_build.log` | the node-specific build of the Rust and C kernels |

plus the Slurm output `hpc_results/slurm-<job name>-<jobid>.out`.

## 6. Send the results

On Niflheim, pack all results:

```bash
tar czf ~/debyecalculator_hpc_results.tgz -C ~/DebyeCalculator hpc_results
```

On your Mac (from the DTU network or VPN), fetch the archive into the repository folder:

```bash
scp <username>@hrid.fysik.dtu.dk:debyecalculator_hpc_results.tgz ~/Downloads/DebyeCalculator/
```

Then tell me it's there. I'll unpack it and run

```bash
python kernels/benchmark_report.py kernels/benchmark_results/apple_m3_max.jsonl hpc_results/*/results.jsonl --out-dir kernels/benchmark_results/all
```

to make the full table and charts across machines and backends.

## Troubleshooting

| problem | what to do |
|---|---|
| `module load Python/3.13.5-GCCcore-14.3.0` fails | `module avail Python/`, then e.g. `export DC_PYTHON_MODULE=Python/3.12.3-GCCcore-13.3.0` (and a `DC_RUST_MODULE` of the same GCCcore) before `setup.sh` and `submit_all.sh` |
| `No environment ...debyecalculator-<arch>` in a job | run `setup.sh` on the login node for that partition (table in step 1) |
| CUDA errors, or `torch.cuda.is_available()` is False on a GPU node | compare the driver's CUDA version in `nvidia-smi` (in `environment.txt`) with the wheels: `export DC_TORCH_INDEX=https://download.pytorch.org/whl/cu121` (or `cu118`) and re-run `setup.sh` |
| `triton kernels: unavailable: ...` | the reason is printed; the GPU job still benchmarks PyTorch CUDA; send me `environment.txt` and `pytest.log` |
| `node-specific kernel build failed` | see `native_build.log`; the job continues with the portable builds from `setup.sh`. `DC_NATIVE_BUILD=0` skips the rebuild |
| GPU jobs stay pending with reason `AssocGrpGRES`, `QOSMaxGRESPerUser` or similar | your group may not be authorized for that GPU partition; ask Niflheim support |
| `sbatch: error: ... output file` | run from the `DebyeCalculator` folder and create `hpc_results/` first (`submit_all.sh` does) |
