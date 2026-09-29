[![pypi](https://img.shields.io/pypi/v/Debyecalculator?label=pypi)](https://pypi.org/project/DebyeCalculator/)
[![Python](https://img.shields.io/badge/python-3.8%E2%80%933.14-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/github/license/FrederikLizakJohansen/DebyeCalculator)]([https://github.com/lfwa/carbontracker/blob/master/LICENSE](https://github.com/FrederikLizakJohansen/DebyeCalculator/blob/main/LICENSE.txt))
[![ChemRxiv](https://img.shields.io/badge/ChemRxiv%20%20-8A2BE2)](https://chemrxiv.org/engage/chemrxiv/article-details/651ec9668bab5d2055b2d009)
[![ReadTheDocs](https://img.shields.io/readthedocs/debyecalculator)](https://debyecalculator.readthedocs.io/en/latest/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.06024/status.svg)](https://doi.org/10.21105/joss.06024)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/DebyeCalculator)](https://pypistats.org/packages/debyecalculator)

<img src="https://raw.githubusercontent.com/FrederikLizakJohansen/DebyeCalculator/main/logo_DebyeCalculator.png" alt="DebyeCalculator" width="350"/>

Welcome to `DebyeCalculator`! This is a simple tool for calculating the scattering intensity $I(Q)$ through the Debye scattering equation, the Total Scattering Structure Function $S(Q)$, the Reduced Total Scattering Function $F(Q)$, and the Reduced Atomic Pair Distribution Function $G(r)$ from an atomic structure. 
The Debye scattering equation can be used to compute the scattering pattern of any atomic structure and is commonly used to study both crystalline and non-crystalline materials with a range of scattering techniques like powder diffraction, total scattering with pair distribution function and small-angle scattering. Although the Debye scattering equation is extremely versatile, the computation of the double sum, which scales O(N<sup>2</sup>), has limited the practical use of the equation.
Here, we provide an optimised code for the calculation of the Debye scattering equation on Graphics processing units (GPUs) which accelerate the calculations with orders of magnitudes.
> For ideal periodic crystals, [BraggCalculator](https://github.com/FrederikLizakJohansen/BraggCalculator) provides a complementary reciprocal-space approach to powder X-ray and neutron diffraction that exploits translational symmetry, while `DebyeCalculator` retains finite-size and non-periodic information through its real-space pair sum.

1. [Installation](#installation)
    1. [Prerequisites](#prerequisites)
    2. [Install with pip](#install-with-pip)
    3. [Install locally](#install-locally)
    4. [Upgrading from 1.0.x](#upgrading-from-10x)
    5. [GPU support](#gpu-support)
3. [Usage](#usage)
    1. [Desktop app](#desktop-app)
    2. [Interactive mode](#interactive-mode)
    3. [Example usage](#example-usage)
3. [Demo](#demo)
4. [Additional implementation details](#additional-implementation-details)
5. [Authors](#authors)
6. [Cite](#cite)
7. [Contributing to the software](#contributing-to-the-software)
    1. [Reporting issues](#reporting-issues)
    2. [Seeking support](#seeking-support)

# Installation

## Prerequisites

`DebyeCalculator` requires Python 3.8–3.14. If needed, create an environment with one of these Python versions:
```bash
conda create -n debyecalculator_env python=3.12
```
```bash
conda activate debyecalculator_env
```

Before installing the `DebyeCalculator` package, ensure that you have PyTorch installed. Follow the instructions on the official PyTorch website to install the appropriate version for your system: [PyTorch Installation Guide](https://pytorch.org/get-started/locally/). 

**NOTE**: Recent PyTorch releases require Python 3.9 or higher; on Python 3.8, pip installs PyTorch 2.4, the last release supporting it.

## Install with [pip](https://pypi.org/project/DebyeCalculator/)

Run the following command to install the __DebyeCalculator__ package. (**Requires**: Python 3.8–3.14)
```
pip install debyecalculator
```

## Install locally

Clone the repo
```
git clone https://github.com/FrederikLizakJohansen/DebyeCalculator.git
```

Run the following command to install the __DebyeCalculator__ package. (**Requires**: Python 3.8–3.14)
```
python -m pip install .
```

### Testing the local installation
To ensure that the installation is set up correctly and your environment is ready to go, we recommend running the included unit tests. </br>
First, make sure you have [pytest](https://docs.pytest.org/en/stable/) installed. If not, you can install it using:
```bash
pip install pytest
```
After installing the package, open a terminal or command prompt and navigate to the root directory of the package. Then run the following command to execute the tests:
```bash
pytest
```

## Upgrading from 1.0.x

Version 1.1 is a minor release in the sense of [semantic versioning](https://semver.org/): code written for 1.0.x runs unchanged and gives the same results within floating-point precision, only faster and with less memory. Python 3.7 is no longer supported; pip keeps Python 3.7 users on 1.0.14.

| Change | Public API | What it means when upgrading |
|---|---|---|
| Function signatures and return types | Unchanged | `iq`, `sq`, `fq`, `gr`, `generate_nanoparticles` and `interact()` take the same arguments and return the same types. The constructor keeps the 1.0.14 parameter order; the new parameters `pair_sum` and `num_threads` are appended with defaults. |
| Python support | Extended; 3.7 dropped | Python 3.8–3.14 (1.0.14: 3.7–3.11). pip reads `Requires-Python: >=3.8,<3.15` and installs the newest compatible release, so users on Python 3.7 keep getting 1.0.14 from `pip install debyecalculator`. |
| Numerical results | Within floating-point precision | The default `pair_sum='grid'` deviates from 1.0.14 by at most ~3e-6 (float32) and ~1e-9 (float64), relative to each function's maximum. `pair_sum='direct'` is the 1.0.x algorithm and reproduces 1.0.14 to ~1e-15 in float64 (~1e-6 in float32, from summation order). |
| Generated nanoparticles | Unchanged | Bit-identical to 1.0.14. |
| `batch_size` | Meaning unchanged, default changed | Still the number of atom pairs per batch, bounding memory. The default `None` now chooses the batch size: 3,000,000 pairs for `pair_sum='grid'`, 10,000 (the 1.0.14 default) for `pair_sum='direct'`. Explicit values are used as given; small ones such as 10,000 run faster than 1.0.14 but about 35× slower than the automatic choice. |
| CPU threads | New behaviour, with opt-out | On the CPU, batches run in a thread pool. While it runs, PyTorch's process-wide thread count is 1; it is restored afterwards, also for overlapping calls from several threads. `num_threads=1` runs sequentially and leaves PyTorch's thread settings untouched. |
| Gradients | Preserved | Results remain differentiable with respect to atomic positions, now at any particle size with memory bounded by `batch_size`. |
| `keep_on_device=False` with positions that require gradients | Relaxed | Returns NumPy arrays; 1.0.14 raised an error. |
| `StructureTuple.triu_indices`, `StructureTuple.unique_inverse` | Internal fields | `None`; the O(N²) index arrays are no longer built. They are only returned by underscore-prefixed methods. |
| Memory | Lower | Bounded by `batch_size`, independent of particle size. |

To reproduce results of 1.0.x exactly, use the 1.0.x algorithm or install the previous release:
```python
calc = DebyeCalculator(pair_sum='direct')   # 1.0.x pair sum; identical to 1.0.14 in float64
```
```bash
pip install debyecalculator==1.0.14
```

## GPU Support

The `DebyeCalculator` package supports GPU acceleration using PyTorch with CUDA. Follow these steps to enable GPU support:

### 1. Verify GPU Availability

After installing PyTorch with CUDA, you can check if your GPU is available by running the following code snippet in a Python environment:

```python
import torch
print("CUDA available:", torch.cuda.is_available())
```

### 2. Specify GPU Device in DebyeCalculator
When creating an instance of DebyeCalculator, you can specify the device as 'cuda' to utilize GPU acceleration:

```python
from debyecalculator import DebyeCalculator

calc = DebyeCalculator(device='cuda')
```

# Usage

## Desktop app

A standalone desktop app (Windows, macOS, Linux) calculates and plots I(Q), S(Q), F(Q) and G(r) live while parameters change. Install it with the `gui` extra (Python 3.8 or later) and start it from a terminal:

```bash
pip install "debyecalculator[gui]"
debyecalculator-gui                      # or: python -m debyecalculator.gui
debyecalculator-gui particle.cif other.xyz
```

- Load several `.cif`, `.xyz` or other ASE-readable files at once (dialog or drag-and-drop); duplicate a CIF to compare particle radii.
- Every parameter updates the plots while a slider moves: particle radius, Q- and r-range, Qstep, Qdamp, B<sub>iso</sub>, r<sub>thres</sub>, Lorch modification, radiation type, partial pair, device and precision.
- Co-plot structures overlaid, stacked with an adjustable offset, or in separate rows; normalise curves and use logarithmic axes.
- Presets for small-angle scattering, powder diffraction and total scattering.
- Show all element-pair partials of a structure as dashed curves.
- Overlay measured I(Q), S(Q), F(Q) or G(r) (two-column text files, I(Q) against Q or 2θ), with a fitted scale, difference curve and Rw.
- I(Q) against 2θ for a chosen wavelength; a cursor readout of all curves; a rotatable 3D view of the particle.
- Calculations longer than a quarter of a second show progress and can be cancelled (Esc).
- Light and dark theme (View → Theme), following the operating system by default.
- Recent files, and the last session and window layout restored at startup.
- Export data (CSV with metadata), figures (PNG, SVG, PDF), generated particles (XYZ), and save or reopen a session.

The notebook widget below (`DebyeCalculator().interact()`) remains available.

## Interactive mode
<b>IMPORTANT: </b> CHANGES TO INTERACTIVE MODE AS OF JANUARY 2024 (DebyeCalculator version >=1.0.5)
In the lastest version of DebyeCalculator, we are unfortunately experiences some issues with Google Colab that does not allow the package to display the <tt>interact()</tt> widget. If you experience any related issues, please refer to this [statement](https://github.com/FrederikLizakJohansen/DebyeCalculator/issues/15#issuecomment-1873764451 'here') for further clarification and workarounds. 

## Example Usage

This section provides quick examples on how to use the `DebyeCalculator` class for generating both total and partial scattering intensities from particle structures defined in `.xyz` files, `.cif` files, or directly from structure tuples.

### Generating Scattering

To calculate the scattering intensity $I(Q)$ for a particle, you can use different structure sources:

```python
from debyecalculator import DebyeCalculator
import torch

# Initialize the DebyeCalculator object
calc = DebyeCalculator(qmin=1.0, qmax=8.0, qstep=0.01)

# Define structure sources
xyz_file = "some_path/some_file.xyz"
cif_file = "some_path/some_file.cif"
structure_tuple = (
    ["Fe", "Fe", "O", "O"],
    torch.tensor(
        [[0.5377, 0.7068, 0.8589],
         [0.1576, 0.1456, 0.8799],
         [0.5932, 0.0204, 0.6759],
         [0.6946, 0.4114, 0.4869]]
    )
)

# Calculate I(Q) from different sources
q, iq_xyz = calc.iq(xyz_file)
q, iq_cif = calc.iq(cif_file)
q, iq_tuple = calc.iq(structure_tuple)
```

### Generating Partial Scattering

DebyeCalculator also allows users to extract the partial scattering for specific pairs of atomic species within a structure:
```python
# Create an instance of DebyeCalculator with appropriate parameters
calc = DebyeCalculator(qmin=1.0, qmax=20.0)

# Load a single particle from a .xyz file and calculate partial I(Q) for specific atom pairs
# Replace 'X' and 'Y' with the atomic symbols present in your structure
q, iq_XX = calc.iq("path/to/nanoparticle.xyz", partial="X-X")
q, iq_YY = calc.iq("path/to/nanoparticle.xyz", partial="Y-Y")
q, iq_XY = calc.iq("path/to/nanoparticle.xyz", partial="X-Y")
```
**Note:** When combining signals from partial scattering, be cautious to avoid double-counting interactions between atoms.

### Gradients with respect to atomic positions

All functions are differentiable with respect to the atomic positions with PyTorch autograd, e.g. to refine a structure against a measured PDF by gradient descent. Pass the positions as a tensor with `requires_grad=True` and keep the results on the device:
```python
import torch

calc = DebyeCalculator(device='cpu', rmax=30.0)
xyz = torch.tensor(positions, requires_grad=True)          # (N, 3) positions in Å
optimizer = torch.optim.Adam([xyz], lr=1e-3)

for step in range(200):
    optimizer.zero_grad()
    r, gr = calc.gr((elements, xyz), keep_on_device=True)   # elements: list of N symbols
    loss = ((gr - gr_measured) ** 2).sum()                  # gr_measured: tensor on the same r-grid
    loss.backward()
    optimizer.step()
```
The gradient is computed analytically, with the same cost and memory limit as the forward calculation (12,581 atoms: 0.35 s forward, 1.5 s forward and backward on a 12-thread laptop CPU). Its accuracy matches the forward calculation: ~1e-6 relative to the exact gradient in float32, ~1e-11 in float64. Without `keep_on_device=True`, results are NumPy arrays without gradient information.

# Demo
For a more detailed demonstration on how `DebyeCalulator` works, including additional examples, please refer to the `Demo.ipynb` notebook available in the repository, or visit [Colab-Demo](https://tinyurl.com/debyedemo)

# Additional implementation details
See the [docs](/docs) folder. 

# Authors
__Frederik L. Johansen__<sup>1</sup><sup>, 2</sup>   
__Andy S. Anker__<sup>1</sup>   
 
<sup>1</sup> Department of Chemistry & Nano-Science Center, University of Copenhagen, Denmark

<sup>2</sup> Department of Computer Science, University of Copenhagen, Denmark

Should there be any questions, desired improvements or bugs please contact us on GitHub or 
through email: __frjo@di.ku.dk__ and __ansoan@dtu.dk__.

# Cite
If you use our code or our results, please consider citing our [paper](https://doi.org/10.21105/joss.06024). Thanks in advance!

```
@article{Johansen_anker2024debye,
title={A GPU-Accelerated Open-Source Python Package for Calculating Powder Diffraction, Small-Angle-, and Total Scattering with the Debye Scattering Equation},
author={Frederik L. Johansen, Andy S. Anker, Ulrik Friis-Jensen, Erik B. Dam, Kirsten M. Ø. Jensen, Raghavendra Selvan},
journal={Journal of Open Source Software},
year={2024},
issn={2475-9066},
issue={94},
url={"https://joss.theoj.org/papers/10.21105/joss.06024"},
doi={10.5281/zenodo.10659528}
```

# Contributing to the software

We welcome contributions to our software! To contribute, please follow these steps:

1. Fork the repository.
2. Make your changes in a new branch.
3. Submit a pull request.

We'll review your changes and merge them if they meet our quality standards, including passing all unit tests. To ensure that your changes pass the unit tests, please run the tests locally before submitting your pull request. You can also view the test results on our GitHub repository using GitHub Actions.

## Reporting issues

If you encounter any issues or problems with our software, please report them by opening an issue on our GitHub repository. Please include as much detail as possible, including steps to reproduce the issue and any error messages you received.

## Seeking support

If you need help using our software, please reach out to us on our GitHub repository. We'll do our best to assist you and answer any questions you have.

# References
<a id="1">[1]</a>
Waasmaier, D., & Kirfel, A. (1995). New analytical scattering-factor functions for free atoms and ions. Acta Crystallographica Section A, 51(3), 416–431. https://doi.org/10.1107/S0108767394013292
