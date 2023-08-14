[ChemRxiv](XXX)  |  [Paper](XXX)

# DebyeCalculator
Welcome to DebyeCalculator! This is a simple tool for XXXX

1. [Usage](#deepStruc-app)
    1. [Google Colab](#google-colab)
    2. [Install locally](#install-locally)
2. [Authors](#authors)
3. [Cite](#cite)
  

# Usage

## Google Colab

## Install locally

Currently __DebyeCalculator__ is not avaible through PyPI or conda so the package needs to be downloaded manually
Run the following command to install the __DebyeCalculator__ package.  
```
pip install .
or
python setup.py install
```

To verify that __DebyeCalculator__ have been installed properly, try calling the help argument.
```
DebyeCalculator --help

>>> usage: POMFinder [-h] -d DATA [-n NYQUIST] [-i QMIN] [-a QMAX] [-m QDAMP] [-f FILE_NAME]       
>>> 
>>> This is a package which takes a directory of PDF files 
>>> or a specific PDF file. It then determines the best structural 
>>> candidates based of a polyoxometalate catalog. Results can
>>> be fitted to the PDF. 
```  
This should output a list of possible arguments for running __DebyeCalculator__ and indicates that it could find the package! 

# Usage
Now that __DebyeCalculator__ is installed and ready to use, let's discuss the possible arguments. The arguments are described in 
greater detail at the end of this section.

| Arg | Description | Default |  
| --- | --- |  --- |  
|  | __Required argument__ | | 
| `-h` or `--help` | Prints help message. |    
| `-n` or `--nyquist` | Is the data nyquist sampled. __bool__ | `-n True`
| `-i` or `--Qmin` | Qmin value of the experimental PDF. __float__ | `-i 0.7` 
| `-a` or `--Qmax` | Qmax value of the experimental PDF. __float__ | `-a 30` 
| `-m` or `--Qdamp` | Qdamp value of the experimental PDF. __float__ | `-m 0.04`
| `-f` or `--file_name` | Name of the output file. __str__ | `-o ''` 
| `-d` or `--data` | A directory of PDFs or a specific PDF file. __str__ | `-d 5` 

For example
```  
POMFinder --data "Experimental_Data/DanMAX_AlphaKeggin.gr" --nyquist "no" --Qmin 0.7 --Qmax 20 --Qdamp 0.02

>>> The 1st guess from the model is:  icsd_427457_1_0.9rscale.xyz
>>> The 2nd guess from the model is:  icsd_427379_0_0.9rscale.xyz
>>> The 3rd guess from the model is:  icsd_281447_0_1.0rscale.xyz
>>> The 4th guess from the model is:  icsd_423775_0_0.9rscale.xyz
>>> The 5th guess from the model is:  icsd_172542_0_1.1rscale.xyz

```  
# Authors
__Frederik L. Johansen__<sup>1</sup><sup>2</sup>   
__Andy S. Anker__<sup>1</sup>   
 
<sup>1</sup> Department of Chemistry & Nano-Science Center, University of Copenhagen, Denmark.
<sup>2</sup> Department of Computer Science, University of Copenhagen, Denmark.

Should there be any question, desired improvements or bugs please contact us on GitHub or 
through email: __frjo@di.ku.dk__ and __andy@chem.ku.dk__.

# Cite
If you use our code or our results, please consider citing our paper. Thanks in advance!

```
@article{Johansen_anker2023debye,
title={A GPU-Accelerated Open-Source Python Package for Rapid Calculation of the Debye Scattering Equation: Applications in Small-Angle Scattering, Powder diffraction, and Total Scattering with Pair Distribution Function Analysis},
author={Frederik L. Johansen, Andy S. Anker, Ulrik Friis-Jensen, Erik B. Dam, Kirsten M. Ø. Jensen, Raghavendra Selvan},
journal={XXXX}
year={2023}}
```
