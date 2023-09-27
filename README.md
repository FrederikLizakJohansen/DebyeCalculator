[ChemRxiv]  |  [Paper]

# DebyeCalculator
Welcome to `DebyeCalculator`! This is a simple tool for calculating the scattering intensity $I(Q)$ through the Debye scattering equation, the Total Scattering Structure Function $S(Q)$, the Reduced Total Scattering Function $F(Q)$, and the Reduced Atomic Pair Distribution Function $G(r)$ from an atomic structure. 
The Debye scattering equation can be used to compute the scattering pattern of any atomic structure and is commonly used to study both crystalline and non-crystalline materials with a range of scattering techniques like powder diffraction, total scattering with pair distribution function and small-angle scattering. Although the Debye scattering equation is extremely versatile, the computation of the double sum, which scales O(N<sup>2</sup>), has limited the practical use of the equation.
Here, we provide an optimised code for the calculation of the Debye scattering equation on Graphics processing units (GPUs) which accelerate the calculations with orders of magnitudes.

1. [Usage](#usage)
    1. [Interactive mode](#interactive-mode-at-google-colab)
    2. [Install locally](#install-locally)
    3. [Example usage](#example-usage)
2. [Authors](#authors)
3. [Cite](#cite)
4. [Contributing to the software](#contributing-to-the-software)
    1. [Reporting issues](#reporting-issues)
    2. [Seeking support](#seeking-support)


# Usage

## Interactive mode at Google Colab
Follow the instructions in our [Interactive Google Colab notebook](https://github.com/FrederikLizakJohansen/DebyeCalculator/blob/main/InteractiveMode_Colab.ipynb) and try to play around. 

## Install locally

Currently __DebyeCalculator__ is not avaible through PyPI or conda so the package needs to be downloaded manually
Run the following command to install the __DebyeCalculator__ package.  
```
pip install .
or
python setup.py install
```

## Example usage
```python
from DebyeCalculator.XXXX import DebyeCalculator

calc = DebyeCalculator()

# Change parameters
calc.update_parameters(XXXXX)

# Calculate Iq
Q, I = calc.iq(XYZ_file)

# Calculate Sq
Q, S = calc.sq(XYZ_file)
.....

```

# Authors
__Frederik L. Johansen__<sup>1</sup><sup>, 2</sup>   
__Andy S. Anker__<sup>1</sup>   
 
<sup>1</sup> Department of Chemistry & Nano-Science Center, University of Copenhagen, Denmark

<sup>2</sup> Department of Computer Science, University of Copenhagen, Denmark

Should there be any questions, desired improvements or bugs please contact us on GitHub or 
through email: __frjo@di.ku.dk__ and __andy@chem.ku.dk__.

# Cite
If you use our code or our results, please consider citing our paper. Thanks in advance!

```
@article{Johansen_anker2023debye,
title={A GPU-Accelerated Open-Source Python Package for Calculating the Debye Scattering Equation: Applications in Powder diffraction, Small-Angle-, and Total Scattering},
author={Frederik L. Johansen, Andy S. Anker, Ulrik Friis-Jensen, Erik B. Dam, Kirsten M. Ø. Jensen, Raghavendra Selvan},
journal={XXXX}
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
