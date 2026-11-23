# Native kernels for the grid pair sum

Optional fused kernels for the most expensive part of DebyeCalculator: the loop over atom pairs in the
grid pair sum (`DebyeCalculator._spread_pairs`, forward) and in its analytic gradient
(`DebyeCalculator._grid_pair_sums_backward`, backward). Everything else (the distance grid, node sums, autograd,
threading, batching, progress callbacks and cancellation) is the existing PyTorch code, so results, gradients and the
public API are unchanged.

There are three implementations of the same per-pair work (a C version of the CPU kernels was benchmarked and
removed: it was no faster than Rust):

| backend | device | package / module | source |
|---|---|---|---|
| Rust | CPU, float32 and float64 | `debyecalculator-kernels` / `debyecalculator_kernels` (PyO3, one abi3 wheel per platform for Python ≥ 3.8) | `kernels/src/lib.rs` |
| Metal | Apple GPU (`device='mps'`), float32 | part of DebyeCalculator: `debyecalculator.metal_pair_sum`, compiled at run time with `torch.mps.compile_shader` (PyTorch ≥ 2.6) | `debyecalculator/metal_pair_sum.py` |
| Triton | NVIDIA GPU (`device='cuda'`), float32 and float64 | part of DebyeCalculator: `debyecalculator.triton_pair_sum`, compiled at run time by Triton (ships with PyTorch's Linux CUDA wheels) | `debyecalculator/triton_pair_sum.py` |

Without the Rust package, without `torch.mps.compile_shader` or Triton, DebyeCalculator runs exactly as before.
The Triton kernels check themselves against a float64 reference on first use and are skipped (PyTorch code) if the
check fails. Running on Niflheim (DTU): see [hpc/niflheim/README.md](../hpc/niflheim/README.md).

## Install

Rust (needs a Rust toolchain, https://rustup.rs):

```bash
pip install maturin
pip install ./kernels          # or, while developing: maturin develop --release -m kernels/Cargo.toml
```

## Choosing the backend

The environment variable `DEBYECALCULATOR_PAIR_BACKEND` selects the backend of the grid pair sum:

| value | CPU | MPS | CUDA |
|---|---|---|---|
| `auto` (default) | Rust if installed, else PyTorch | Metal if available, else PyTorch | Triton if available, else PyTorch |
| `rust` | Rust (PyTorch if it is not installed) | PyTorch | PyTorch |
| `metal` | PyTorch | Metal (PyTorch if not available) | PyTorch |
| `triton` | PyTorch | PyTorch | Triton (PyTorch if not available) |
| `torch` | PyTorch | PyTorch | PyTorch |

The kernels are used only where they apply: `pair_sum='grid'` (the default); CPU or CUDA with float32 or float64; MPS
with float32 and non-negative occupancies. The direct pair sum and the small structures that use it are unchanged.

## What the kernels do

Per batch of atom pairs (the same batches as the PyTorch code, run in the same thread pool with the GIL released),
each kernel makes one pass over the pairs instead of a sequence of PyTorch operations that each write a batch-sized
temporary tensor (distances, bins, moments, masks, interpolation weights...):

- forward: distance → bin and fractional position s → add the moments w, w·s, w·s², w·s³ to the bin histogram;
- backward: distance → cubic interpolation of the node tables h and k → accumulate the gradients with respect to
  positions and occupancies.

The per-pair arithmetic is done in the calculation dtype in the same operation order as the PyTorch code (no fused
multiply-add: Rust never contracts, the C code is compiled with `-ffp-contract=off`), and accumulated in float64.
The only difference from the PyTorch code is the order of the additions, so:

- float64: results and gradients agree with the PyTorch code to ~1e-14;
- float32: results agree to float32 round-off (~1e-6 relative); measured against float64, the error of the kernels is
  the same as or smaller than that of the PyTorch code (they accumulate in float64 rather than per-batch in float32).

Invalid input (atom indices out of range, distances outside the grid, NaN positions, wrong dtypes or shapes) raises
`ValueError`/`TypeError` instead of reading or writing out of bounds.

### Metal (MPS)

- forward: one GPU thread per row atom and column segment adds the four moments of each pair to the histogram with
  atomic adds, in 64-bit fixed point (two 32-bit atomics with carry): each moment is rounded once (relative 2^-31 of
  the largest pair weight) and the sums are exact, however many pairs fall into a bin. Global atomic throughput limits
  the forward pass (~3–4·10^9 pairs/s on an M3 Max; without the atomics the kernel runs at ~7·10^10 pairs/s).
- backward: one thread per atom and column segment sums the gradient terms of its own atom over ordered pairs, with
  compensated float32 summation, and makes one atomic add per thread: no per-pair atomics.
- The shader is compiled without fast math or contraction. The node sums and gradient tables before and after the
  kernels are the existing PyTorch code, in float32 on MPS; they set the accuracy (~1e-5 relative, like the PyTorch
  MPS path).

### Triton (CUDA)

- forward: each program takes a tile of 32 × 64 atom pairs, and adds the four moments of the valid pairs to the
  float64 histogram with float64 atomic adds (native on all GPUs since compute capability 6.0).
- backward: each program sums the gradient terms of its 32 row atoms over the 64 columns of its tile (ordered pairs)
  and makes one float64 atomic add per row atom: no per-pair atomics.
- Square roots and float32 divisions are correctly rounded (`sqrt_rn`, `div_rn`); tile sizes can be tuned with
  `DEBYECALCULATOR_TRITON_BLOCK_R`, `..._BLOCK_C` and `..._NUM_WARPS`.
- Tested and benchmarked on an NVIDIA A100 and an RTX 3090 (see [BENCHMARK.md](BENCHMARK.md)).

## Tests and benchmark

```bash
pytest debyecalculator                                   # all tests, with the default backend
DEBYECALCULATOR_PAIR_BACKEND=torch pytest debyecalculator  # all tests on the PyTorch code
pytest debyecalculator/test_pair_sum.py -k native        # the Rust kernels against PyTorch
python kernels/benchmark.py --out results.jsonl          # time and peak memory, all backends and precisions
```

See [BENCHMARK.md](BENCHMARK.md) for results.
