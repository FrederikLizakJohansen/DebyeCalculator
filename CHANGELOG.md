# Changelog

## 1.1.1

### Added
- The 3D structure viewer can show the selected nanoparticle independently of plot visibility, or show input,
  primitive, conventional, Niggli-reduced and LLL-reduced unit cells with lattice outlines and boundary atoms.
- The 3D viewer has a center control, and its Structures-tab button toggles between showing and hiding the view.
- Particles above 10,000 atoms use a clearly labelled performance mode that renders an 8,000-atom outer shell;
  scattering calculations and particle exports continue to use every atom.

### Fixed
- Structure and experimental-data lists use the available vertical space to show more imported files.
- Long cursor readouts, status messages, particle labels and structure names no longer increase the GUI window width. Cursor readouts are elided on screen and remain available in full as a tooltip.
- Unit-cell edges are depth-layered so the rear edges pass behind atoms and the front edges pass over them.
- 3D rotation redraws, rapidly changed structure selections and multi-file imports do less duplicate work.
- Closing the GUI during a long structure operation waits for the worker to finish safely.

## 1.1.0

### Upgrading from 1.0.x
Code written for 1.0.x runs unchanged: function signatures, return types and the constructor's parameter order are the same, generated nanoparticles are bit-identical, and results agree with 1.0.14 within floating-point precision. `pair_sum='direct'` reproduces 1.0.14 to ~1e-15 in float64. Python 3.7 is no longer supported; pip reads `Requires-Python: >=3.8,<3.15` and keeps Python 3.7 users on 1.0.14. The README section "Upgrading from 1.0.x" lists every change in behaviour.

### Performance
- The Debye sum spreads pair weights onto a distance grid with cubic Lagrange interpolation and evaluates the sum on the grid nodes. G(r) for 3,833 atoms takes 0.07 s on a 12-thread laptop CPU (10 s in 1.0.14); 190,000 atoms take about 2 minutes.
- Calculation memory is bounded by `batch_size` and independent of particle size.
- Nanoparticle generation uses a k-d tree bond search and vectorised supercell construction (97,000 atoms: 67 s → 1 s). Generated particles are identical to 1.0.14.

### Gradients
- I(Q), S(Q), F(Q) and G(r) are differentiable with respect to atomic positions at any particle size. The backward pass of the pair sum is computed analytically on the distance grid, with the same memory bound as the forward pass. In 1.0.14, autograd worked through the direct pair sum but stored every pair × Q intermediate (about 1 GB per 1.5 million pairs in float32).
- Results converted to NumPy (`keep_on_device=False`) are detached, so positions with `requires_grad=True` no longer raise an error there.

### Changed
- Supports Python 3.8–3.14, numpy 2 and ase ≥ 3.23 (tested with Python 3.8, 3.10, 3.13 and 3.14, numpy up to 2.5, ase up to 3.29). Python 3.7 is no longer supported; pip keeps Python 3.7 users on 1.0.14 (`Requires-Python: >=3.8,<3.15`).
- Dependency versions have lower bounds only; `prettytable` is no longer pinned to 3.0.0.
- `batch_size` still counts atom pairs per batch (shared between CPU threads). Its default `None` chooses 3,000,000 pairs for the grid pair sum and 10,000 for the direct pair sum; in 1.0.14 the default was 10,000 and `None` meant 4,000.
- On the CPU, the pair sum runs in a thread pool; PyTorch's process-wide thread count is 1 while it runs and is restored afterwards, also for overlapping calls.
- `StructureTuple.triu_indices` and `StructureTuple.unique_inverse` are `None`; pairs are enumerated per batch.
- Results of the default grid pair sum differ from 1.0.14 by at most ~3e-6 (float32) and ~1e-9 (float64) relative to each function's maximum.

### Fixed
- `.xyz` files with an occupancy column (five columns) load.
- Nanoparticle generation works with ase ≥ 3.23 (`Atoms.center` is called with a point); particles are identical across ase versions.
- An unsupported structure type raises `TypeError` instead of `NameError` when pymatgen is not installed.
- The benchmark utility locates its reference files without the deprecated `pkg_resources`.

### Added
- `pair_sum='direct'`: the 1.0.x pair sum, for exact reproduction of earlier results.
- `num_threads`: number of CPU threads for the pair sum; `num_threads=1` leaves PyTorch's thread settings untouched.
- Desktop app (`pip install "debyecalculator[gui]"`, `debyecalculator-gui`): live-updating I(Q), S(Q), F(Q) and G(r) for several structures, co-plotting modes, element-pair partials, comparison with experimental data, 2θ axis, 3D particle view, light and dark theme, and export of data, figures, particles and sessions.
- `DebyeCalculator.progress_callback` and `CalculationCancelled` for progress reporting and cancellation of long calculations.
- CI on Python 3.8–3.14 (Linux) and 3.10/3.14 (Windows, macOS).
- `paper/benchmark/run_benchmark.py`: timing and pattern comparison of a release against the current source on CPU and GPU.
