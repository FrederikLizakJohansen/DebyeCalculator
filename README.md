[ChemRxiv](https://chemrxiv.org/engage/chemrxiv/article-details/651ec9668bab5d2055b2d009)  |  [Paper]

![DebyeCalculator](logo.png)

# DebyeCalculator
Welcome to `DebyeCalculator`! This is a simple tool for calculating the scattering intensity $I(Q)$ through the Debye scattering equation, the Total Scattering Structure Function $S(Q)$, the Reduced Total Scattering Function $F(Q)$, and the Reduced Atomic Pair Distribution Function $G(r)$ from an atomic structure. 
The Debye scattering equation can be used to compute the scattering pattern of any atomic structure and is commonly used to study both crystalline and non-crystalline materials with a range of scattering techniques like powder diffraction, total scattering with pair distribution function and small-angle scattering. Although the Debye scattering equation is extremely versatile, the computation of the double sum, which scales O(N<sup>2</sup>), has limited the practical use of the equation.
Here, we provide an optimised code for the calculation of the Debye scattering equation on Graphics processing units (GPUs) which accelerate the calculations with orders of magnitudes.

1. [Installation](#installation)
    1. [Install with pip](#install-with-pip)
    2. [Install locally](#install-locally)
2. [Usage](#usage)
    1. [Interactive mode](#interactive-mode-at-google-colab)
    3. [Example usage](#example-usage)
3. [High-level implementation details](#high-level-implementation-details)
4. [Authors](#authors)
5. [Cite](#cite)
6. [Contributing to the software](#contributing-to-the-software)
    1. [Reporting issues](#reporting-issues)
    2. [Seeking support](#seeking-support)

# Installation

## Install with [pip](https://pypi.org/project/DebyeCalculator/)

Run the following command to install the __DebyeCalculator__ package. (**Requires**: Python >=3.9, <3.12)
```
pip install DebyeCalculator
```

## Install locally

Run the following command to install the __DebyeCalculator__ package.  
```
pip install .
or
python setup.py install
```

# Usage

## Interactive mode at Google Colab
Follow the instructions in our [Interactive Google Colab notebook](https://github.com/FrederikLizakJohansen/DebyeCalculator/blob/main/InteractiveMode_Colab.ipynb) and try to play around. 

## Example usage
```python
from debyecalculator import DebyeCalculator

# Initialise calculator object
calc = DebyeCalculator(qmin=1.0, qmax=8.0, qstep=0.01)

# Define structure path
XYZ_path = "some_path/some_file.xyz"
CIF_path = "some_path/some_file.cif"

# Print object to view all parameters
print(calc)
## [OUTPUT] DebyeCalculator{'qmin': 1.0, 'qmax': 8.0, 'qstep': 0.01, 'rmin': 0.0, 'rmax': 20.0, ...}

# Calculate Powder (X-ray) Diffraction from XYZ-file
Q, I = calc.iq(structure_path=XYZ_path)

# Calculate Powder (X-ray) Diffraction from CIF
Q, I = calc.iq(structure_path=CIF_path, radii=5) # radii is the radius of the particle in Å 

# Update parameters for Small Angle (Neutron) Scattering
calc.update_parameters(qmin=0.0, qmax=3.0, qstep=0.01, radiation_type="neutron")

# Calculate Small Angle (Neutron) Scattering
calc.iq(structure_path=XYZ_path)

# Update parameters for Total Scattering with Pair Distribution Function analysis
calc.update_parameters(qmin=1.0, qmax=30.0, qstep=0.1, radiation_type="xray")

# Calculate Pair Distribution Function
r, G = calc.gr(structure_path=XYZ_path)
.....

```

# High-level implementation details
The core functionality of ´DebyeCalculator´, represented in the following high-level outline, starts with an initialisation function that sets user-defined parameters or sets them to default. They include scattering parameters (such as Q-range, Q-step, PDF r-range and r-step, atomic vibrations, radiation type, and instrumental parameters) and hardware parameters. During this initialisation phase, the calculation of the atomic form factors (for X-ray) or scattering lengths (for neutrons) is prepared based on the radiation type. Once initialised, ´DebyeCalculator´ can compute various quantities: the scattering intensity $I(Q)$ through the Debye scattering equation, the Total Scattering Structure Function $S(Q)$, the Reduced Total Scattering Function $F(Q)$, and the Reduced Atomic Pair Distribution Function $G(r)$. In this section, we specifically outline the $G(r)$ calculation using X-ray scattering. This is because the process for calculating $G(r)$ inherently involves the calculations for $I(Q)$, S(Q), and F(Q). When calling the `gr` function, ´DebyeCalculator´ loads the structure and computes the atomic form factors[[1]](#1). Following this, it calculates the scattering intensity $I(Q)$ using the Debye scattering equation and subsequently determines the structure factor $S(Q)$ and $F(Q)$. Necessary modifications, such as dampening and Lorch modifications, are applied before computing the $G(r)$. ´DebyeCalculator´ outputs the calculated functions to the CPU by default to allow for immediate analysis of the results, but users have the flexibility to retain the output on the GPU. 
It is worth noting that the majority of the compute time is spent on the double sum calculation in the Debye scattering equation. This is where GPU acceleration can enhance performance compared to single core CPUs. For all atom pairs, intermediate products of distances, form factors, and momentum transfers need to be calculated and stored temporarily. Calculating the intermediate products is computationally inexpensive but demands significant memory. This restricts the ability to apply the Debye scattering equation to structures with an increasing number of atoms. The batching schema in ´DebyeCalculator´ aims to mitigate these memory requirements by breaking down the calculations into smaller chunks that fit into the available GPU memory, thus enabling the calculation of scattering intensities for structures with a large number of atoms. The trade-off is a slight increase in computation time. Users with more substantial GPU memory can accommodate large structures while maintaining high computation speeds.

```plaintext
CLASS ´DebyeCalculator´:                                                  
  FUNCTION Initialise(parameters...):
      - Set class parameters based on given input or defaults           
      - Setup scattering parameters (e.g., Q-values, r-values) and hardware parameters  
      - Load atomic form factor coefficients                             
      - Setup form factor calculation based on radiation type           
  
  FUNCTION gr(structure_path, keep_on_device=False):                
      - Load atomic structure from given structure_path                       
      - Calculate atomic form factors                                
      - Calculate scattering intensity I(Q) (Debye scattering equation) 
      - Compute structure factor S(Q) based on I(Q)                     
      - Calculate F(Q) based on Q-values and S(Q)                       
      - Apply modifications if necessary (like dampening and Lorch)       
      - Calculate pair distribution function G(r) based on F(Q)         
      - Return G(r) either on GPU or CPU            
```

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

