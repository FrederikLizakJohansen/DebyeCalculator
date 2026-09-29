# Benchmark: release vs. current source

`run_benchmark.py` times G(r) for spherical AntiFluorite Co2O particles with an earlier release (default `v1.0.14`) and with the current source tree, on the CPU and, when PyTorch sees one, on the CUDA GPU. It also compares the calculated I(Q), S(Q), F(Q) and G(r) of both versions.

```bash
python paper/benchmark/run_benchmark.py
```

The earlier release is checked out with `git worktree`, and both versions run in the active Python environment, which needs the package dependencies (`ase<3.23`, `torch`, `matplotlib`, `scipy`, ...).

Output in `paper/benchmark/benchmark_output/`:

| File | Content |
|---|---|
| `figure_benchmark_time.(png\|pdf)` | Calculation time against particle diameter and number of atoms |
| `figure_benchmark_speedup.(png\|pdf)` | Speed-up of the current source over the release, per device |
| `figure_benchmark_memory.(png\|pdf)` | Peak memory: allocated GPU memory, or process RSS on the CPU |
| `figure_accuracy.(png\|pdf)` | Patterns of both versions and their differences to the release in float64 |
| `summary.md` | Hardware, timing table and maximum pattern differences |
| `results.json` | All measurements |

Each measurement runs in its own process. A series stops after the first structure that takes longer than `--time-limit` seconds (default 120) or fails, for example by running out of memory. Re-running the command skips finished measurements, and `--plot-only` redraws the figures from `results.json`.

Useful options: `--versions new` (skip the release), `--radii 2:40:2,45:80:5`, `--devices cpu cuda`, `--old-ref <git ref>`, `--repetitions 5`, `--dtype float64`, `--batch-size <pairs>`. See `--help`.
