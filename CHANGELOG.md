# Changelog

## 1.1.0

### Upgrading from 1.0.x
Code written for 1.0.x runs unchanged: function signatures, return types and the constructor's parameter order are the same, Python support is unchanged, generated nanoparticles are bit-identical, and results agree with 1.0.14 within floating-point precision. `pair_sum='direct'` reproduces 1.0.14 to ~1e-15 in float64. The README section "Upgrading from 1.0.x" lists every change in behaviour.

### Performance
- The Debye sum spreads pair weights onto a distance grid with cubic Lagrange interpolation and evaluates the sum on the grid nodes. G(r) for 3,833 atoms takes 0.07 s on a 12-thread laptop CPU (10 s in 1.0.14); 190,000 atoms take about 2 minutes.
- Calculation memory is bounded by `batch_size` and independent of particle size.
- Nanoparticle generation uses a k-d tree bond search and vectorised supercell construction (97,000 atoms: 67 s → 1 s). Generated particles are identical to 1.0.14.

### Gradients
- I(Q), S(Q), F(Q) and G(r) are differentiable with respect to atomic positions at any particle size. The backward pass of the pair sum is computed analytically on the distance grid, with the same memory bound as the forward pass. In 1.0.14, autograd worked through the direct pair sum but stored every pair × Q intermediate (about 1 GB per 1.5 million pairs in float32).
- Results converted to NumPy (`keep_on_device=False`) are detached, so positions with `requires_grad=True` no longer raise an error there.

### Changed
- `batch_size` still counts atom pairs per batch (shared between CPU threads). Its default `None` chooses 3,000,000 pairs for the grid pair sum and 10,000 for the direct pair sum; in 1.0.14 the default was 10,000 and `None` meant 4,000.
- On the CPU, the pair sum runs in a thread pool; PyTorch's process-wide thread count is 1 while it runs and is restored afterwards, also for overlapping calls.
- `StructureTuple.triu_indices` and `StructureTuple.unique_inverse` are `None`; pairs are enumerated per batch.
- Results of the default grid pair sum differ from 1.0.14 by at most ~3e-6 (float32) and ~1e-9 (float64) relative to each function's maximum.

### Fixed
- `.xyz` files with an occupancy column (five columns) load.

### Added
- `pair_sum='direct'`: the 1.0.x pair sum, for exact reproduction of earlier results.
- `num_threads`: number of CPU threads for the pair sum; `num_threads=1` leaves PyTorch's thread settings untouched.
- `paper/benchmark/run_benchmark.py`: timing and pattern comparison of a release against the current source on CPU and GPU.
