[![pypi](https://img.shields.io/pypi/v/Debyecalculator?label=pypi)](https://pypi.org/project/DebyeCalculator/)
[![Python](https://img.shields.io/badge/python-%3E%3D3.7-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/github/license/FrederikLizakJohansen/DebyeCalculator)]([https://github.com/lfwa/carbontracker/blob/master/LICENSE](https://github.com/FrederikLizakJohansen/DebyeCalculator/blob/main/LICENSE.txt))
[![ChemRxiv](https://img.shields.io/badge/ChemRxiv%20%20-8A2BE2)](https://chemrxiv.org/engage/chemrxiv/article-details/651ec9668bab5d2055b2d009)
[![ReadTheDocs](https://img.shields.io/readthedocs/debyecalculator)](https://debyecalculator.readthedocs.io/en/latest/)
![DebyeCalculator](https://raw.githubusercontent.com/FrederikLizakJohansen/DebyeCalculator/main/logo.png)

# DebyeCalculator
Welcome to `DebyeCalculator`! This is a simple tool for calculating the scattering intensity $I(Q)$ through the Debye scattering equation, the Total Scattering Structure Function $S(Q)$, the Reduced Total Scattering Function $F(Q)$, and the Reduced Atomic Pair Distribution Function $G(r)$ from an atomic structure. 
The Debye scattering equation can be used to compute the scattering pattern of any atomic structure and is commonly used to study both crystalline and non-crystalline materials with a range of scattering techniques like powder diffraction, total scattering with pair distribution function and small-angle scattering. Although the Debye scattering equation is extremely versatile, the computation of the double sum, which scales O(N<sup>2</sup>), has limited the practical use of the equation.
Here, we provide an optimised code for the calculation of the Debye scattering equation on Graphics processing units (GPUs) which accelerate the calculations with orders of magnitudes.

1. [Installation](#installation)
    1. [Prerequisites](#prerequisites)
    2. [Install with pip](#install-with-pip)
    3. [Install locally](#install-locally)
    5. [GPU support](#gpu-support)
3. [Usage](#usage)
    1. [Interactive mode](#interactive-mode-at-google-colab)
    2. [Example usage](#example-usage)
    3. [Partial Gr](#partial-gr)
4. [Additional implementation details](#additional-implementation-details)
5. [Authors](#authors)
6. [Cite](#cite)
7. [Contributing to the software](#contributing-to-the-software)
    1. [Reporting issues](#reporting-issues)
    2. [Seeking support](#seeking-support)

# Installation

## Prerequisites

`DebyeCalculator` requires Python version >=3.7, <3.12. If needed, create an environment with any of these Python versions:
```bash
conda create -n debyecalculator_env python=3.9
```
```bash
conda activate debyecalculator_env
```

Before installing the `DebyeCalculator` package, ensure that you have PyTorch installed. Follow the instructions on the official PyTorch website to install the appropriate version for your system: [PyTorch Installation Guide](https://pytorch.org/get-started/locally/). 

**NOTE**: Installing an [earlier version](https://pytorch.org/get-started/previous-versions/) of PyTorch (<=1.13.1) will be necessary if you're running Python 3.7, since the latest PyTorch version requires Python 3.8 or higher.

## Install with [pip](https://pypi.org/project/DebyeCalculator/)

Run the following command to install the __DebyeCalculator__ package. (**Requires**: Python >=3.7, <3.12)
```
pip install debyecalculator
```

## Install locally

Clone the repo
```
git clone https://github.com/FrederikLizakJohansen/DebyeCalculator.git
```

Run the following command to install the __DebyeCalculator__ package. (**Requires**: Python >=3.7, <3.12)
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

## Interactive mode at Google Colab
Follow the instructions in our [Interactive Google Colab notebook](https://github.com/FrederikLizakJohansen/DebyeCalculator/blob/main/InteractiveMode_Colab.ipynb) and try to play around. 

<b>IMPORTANT: </b> CHANGES TO INTERACTIVE MODE AS OF JANUARY 2024 (DebyeCalculator version >=1.0.5)
In the lastest version of DebyeCalculator, we are unfortunately experiences some issues with Google Colab that does not allow the package to display the <tt>interact()</tt> widget. If you experience any related issues, please refer to this [statement](https://github.com/FrederikLizakJohansen/DebyeCalculator/issues/15#issuecomment-1873764451 'here') for further clarification and workarounds. 

## Example usage
```python
from debyecalculator import DebyeCalculator
import torch

# Initialise calculator object
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
))

# Print object to view all parameters
print(calc)
## [OUTPUT] DebyeCalculator{'qmin': 1.0, 'qmax': 8.0, 'qstep': 0.01, 'rmin': 0.0, 'rmax': 20.0, ...}

# Calculate Powder (X-ray) Diffraction from XYZ-file
Q, I = calc.iq(structure_source=xyz_file)

# Calculate Powder (X-ray) Diffraction from CIF
Q, I = calc.iq(structure_source=cif_file, radii=5) # radii is the radius of the particle in Å

# Calculate Powder (X-ray) Diffraction from Tuple
Q, I = calc.iq(structure_source=structure_tuple)

# Update parameters for Small Angle (Neutron) Scattering
calc.update_parameters(qmin=0.0, qmax=3.0, qstep=0.01, radiation_type="neutron")

# Calculate Small Angle (Neutron) Scattering
Q, I = calc.iq(structure_source=xyz_file)

# Update parameters for Total Scattering with Pair Distribution Function analysis
calc.update_parameters(qmin=1.0, qmax=30.0, qstep=0.1, radiation_type="xray")

# Calculate Pair Distribution Function
r, G = calc.gr(structure_source=xyz_file)
.....

```

## Partial Gr
```
# Create a mask for oxygen atoms
mask = [atom == "O" for atom in structure_tuple[0]]

# Apply the mask to the structure_tuple
structure_tuple = (
    [atom for atom, m in zip(structure_tuple[0], mask) if m],
    structure_tuple[1][mask]
)

# Calculate Pair Distribution Function
r_partial, Gr_partial = calc.gr(structure_source=structure_tuple)
```

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
If you use our code or our results, please consider citing our paper. Thanks in advance!

```
@article{Johansen_anker2023debye,
title={A GPU-Accelerated Open-Source Python Package for Calculating Powder Diffraction, Small-Angle-, and Total Scattering with the Debye Scattering Equation},
author={Frederik L. Johansen, Andy S. Anker, Ulrik Friis-Jensen, Erik B. Dam, Kirsten M. Ø. Jensen, Raghavendra Selvan},
journal={ChemRxiv}
year={2023}}
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

