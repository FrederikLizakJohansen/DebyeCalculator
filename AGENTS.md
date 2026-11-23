# Notes for AI coding agents

DebyeCalculator computes I(Q), S(Q), F(Q) and G(r) of atomic structures with the Debye scattering equation, in
PyTorch, on CPU, CUDA and Apple GPUs (MPS). Users rely on it for gradients (structure refinement), so speed,
memory, accuracy and differentiability all matter.

## Where things are

| Path | What |
|---|---|
| `debyecalculator/debye_calculator.py` | The `DebyeCalculator` class: public `iq`, `sq`, `fq`, `gr`; pair sums; backend selection |
| `debyecalculator/metal_pair_sum.py` | Metal kernels (MPS, float32), compiled at run time with `torch.mps.compile_shader` |
| `debyecalculator/triton_pair_sum.py` | Triton kernels (CUDA, float32/float64), compiled at run time |
| `kernels/` | Rust kernels (CPU), a separate PyO3 package `debyecalculator-kernels` / module `debyecalculator_kernels` |
| `kernels/README.md`, `kernels/BENCHMARK.md` | How the kernels work; benchmarks on all machines |
| `debyecalculator/utility/generate.py` | `generate_nanoparticles` |
| `debyecalculator/gui/` | Desktop app (`debyecalculator-gui`, `gui` extra) |
| `debyecalculator/test_*.py` | Tests (pytest) |

## How a calculation runs

- `pair_sum='grid'` (default) is the main algorithm. It spreads pair weights onto a distance grid with cubic
  Lagrange interpolation, then evaluates the sum on the grid nodes. The backward pass is analytic: the gradient
  with respect to positions and occupancies is computed on the same grid with the same memory bound.
- `pair_sum='direct'` is the 1.0.x algorithm (exact, every pair at every Q). Small structures use it automatically.
- The loop over atom pairs (forward: `_spread_pairs`; backward: `_grid_pair_sums_backward`) has four
  implementations: PyTorch, Rust, Metal and Triton. Everything around them (grid, node sums, autograd, batching,
  thread pool, progress callbacks, cancellation) is shared PyTorch code.
- How the backend is chosen:
  - The environment variable `DEBYECALCULATOR_PAIR_BACKEND` sets it: `auto` (default), `rust`, `metal`, `triton` or
    `torch`.
  - Tests set it per instance with `calc._pair_backend`.
  - `auto` uses the kernel for the device when it is available, and PyTorch otherwise.
- Where each kernel applies:
  - Rust: CPU, float32 and float64.
  - Triton: CUDA, float32 and float64. It checks itself against float64 on first use and falls back to PyTorch if
    the check fails.
  - Metal: MPS, float32, non-negative occupancies.

## Rules

- **Public API is stable.** The function signatures and return types of `iq`, `sq`, `fq`, `gr`,
  `generate_nanoparticles` and `interact()` stay the same. New constructor parameters go at the end, with
  defaults. The README section "Upgrading from 1.0.x" lists every behaviour change; keep it accurate.
- **All backends must agree.** If you change the per-pair arithmetic, change all four implementations, in the
  same operation order, without fused multiply-add. Agreement with the PyTorch code:
  - float64: ~1e-14.
  - float32: round-off.
- **Gradients are a feature.** Every change to a calculation must keep gradients with respect to positions (and
  occupancies) correct. Test them against finite differences or against the PyTorch path.
- **Memory must not grow with pairs × Q or atoms × Q.** Calculation memory is bounded by `batch_size`. Don't build
  per-pair or per-atom-per-Q tensors.
- **Python 3.8–3.14 are supported.** Don't use syntax newer than 3.8 (no `match`, no `X | Y` type unions, no
  `list[int]` in annotations evaluated at run time). The Rust package targets abi3, Python 3.8 and later.
- **Optional dependencies stay optional.** Without the Rust package, without `torch.mps.compile_shader` or without
  Triton, everything must run on the PyTorch code.
- Record user-visible changes in `CHANGELOG.md` under "Unreleased".

## Testing

```bash
pip install ./kernels                                       # Rust kernels (needs https://rustup.rs); or:
maturin develop --release -m kernels/Cargo.toml             # while developing
pytest debyecalculator                                      # with the default backend
DEBYECALCULATOR_PAIR_BACKEND=torch pytest debyecalculator   # on the PyTorch code
pytest debyecalculator/test_pair_sum.py                     # pair sums, kernels, gradients, memory
```

- Metal tests run only on Apple silicon. Triton tests run only on an NVIDIA GPU.
- CI builds the Rust kernels, so they are tested there on Linux, macOS and Windows.

## Benchmarking

Use `kernels/benchmark.py` (report: `kernels/benchmark_report.py`). For numbers you can compare:

- Run every configuration in a fresh process.
- Warm up on a tiny particle first, to pay for kernel compilation and library loading.
- Report the fastest of several runs.
- Report peak memory above the level before the run: RSS sampled every millisecond, plus GPU tensor memory.

Results are in `kernels/benchmark_results/`.
