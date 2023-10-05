---
title: 'A GPU-Accelerated Open-Source Python Package for Calculating Powder Diffraction, Small-Angle-, and Total Scattering with the Debye Scattering Equation'
tags:
  - Python
  - Scattering
  - Debye Scattering Equation
  - GPU-accelerated
  - Nanoparticles
  - Pair Distribution Function Analysis
authors:
  - name: Frederik L. Johansen
    orcid: 0000-0002-8049-8624
    equal-contrib: true
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Andy S. Anker
    orcid: 0000-0002-7403-6642
    corresponding: true # (This is how to denote the corresponding author)
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 1
  - name: Ulrik Friis-Jensen
    orcid: 0000-0001-6154-1167
    affiliation: "1, 2"
  - name: Erik B. Dam
    orcid: 0000-0002-8888-2524
    affiliation: 2
  - name: Kirsten M. Ø. Jensen
    orcid: 0000-0003-0291-217X
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 1
  - name: Raghavendra Selvan
    orcid: 0000-0003-4302-0207
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: "2, 3"

affiliations:
 - name: Department of Chemistry & Nano-Science Center, University of Copenhagen, Denmark
   index: 1
 - name: Department of Computer Science, University of Copenhagen, Denmark
   index: 2
 - name: Department of Neuroscience, University of Copenhagen, Denmark
   index: 3
date: 5 October 2023
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
# aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
# aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

The Debye scattering equation, derived in 1915 by Peter Debye, is used to calculate scattering intensities from atomic structures considering the position of each atom in the structure[@debye:1915; @scardi:2016]:

\begin{equation}\label{eq:Debye}
I(Q) = \sum_{\nu=1}^{N} \sum_{\mu=1}^{N} b_{\nu} b_{\mu} \frac{\sin(Qr_{\nu\mu})}{Qr_{\nu\mu}}
\end{equation}

In this equation $Q$ is the momentum transfer of the scattered radiation, $N$ is the number of atoms in the structure, and $r_{\nu\mu}$ is the distance between atoms $\nu$ and $\mu$. For X-ray radiation, the atomic form factor, $b$, depends strongly on $Q$ and is usually denoted as $f(Q)$, but for neutrons, $b$ is independent of $Q$ and referred to as the scattering length. 
The Debye scattering equation can be used to compute the scattering pattern of any atomic structure and is commonly used to study both crystalline and non-crystalline materials with a range of scattering techniques like powder diffraction (PD), total scattering (TS) with pair distribution function (PDF) analysis, and small-angle scattering (SAS)[@scardi:2016]. Although the Debye scattering equation is extremely versatile, the computation of the double sum, which scales $O(N^{2})$, has limited the practical use of the equation.

With the advancement in computer hardware[@schaller1997moore], analysis of larger structures is now feasible using the Debye scattering equation. Modern central processing units (CPUs), ranging from tens to hundreds of cores offer an opportunity to parallelise computations, significantly enhancing compute efficiency. The same goes for graphics processing units (GPUs), which are designed with multiple cores acting as individual accelerated processing units that can work on different tasks simultaneously. In contrast, CPUs usually have fewer cores optimised for more general-purpose computing. This means that a GPU can execute multiple simple instructions in parallel, while a CPU might handle fewer parallel tasks[@garland2008parallel]. Therefore, GPUs are better suited for calculations such as the Debye scattering equation, where many computations can be performed simultaneously. Taking advantage of such GPU acceleration could yield computational speeds that surpass those of even the most advanced multi-core CPUs; by orders of magnitude. We introduce a GPU-accelerated open-source Python package, named ´DebyeCalculator´, for rapid calculation of the Debye scattering equation from chemical structures represented as xyz-files or CIF-files. The xyz-format is commonly used in materials chemistry for the description of discrete particles and simply consists of a list of atomic identities and their respective Cartesian coordinates (x, y, and z). ´DebyeCalculator´ can take a crystallographic information file (CIF) and a user-defined spherical radius as input to generate an xyz-file from which a scattering pattern is calculated. We further calculate the PDF as described by Egami and Billinge[@egami2003underneath]. We show that our software can simulate PD, TS, SAS, and PDF data orders of magnitudes faster than DiffPy-CMI[@juhas2015complex]. ´DebyeCalculator´ is an open-source project that is readily available through GitHub (https://github.com/FrederikLizakJohansen/DebyeCalculator) and PyPI (https://pypi.org/project/DebyeCalculator/).

The core functionality of ´DebyeCalculator´, represented in the following high-level outline, starts with an initialisation function that sets user-defined parameters or sets them to default. They include scattering parameters (such as Q-range, Q-step, PDF r-range and r-step, atomic vibrations, radiation type, and instrumental parameters) and hardware parameters. During this initialisation phase, the calculation of the atomic form factors (for X-ray) or scattering lengths (for neutrons) is prepared based on the radiation type. Once initialised, ´DebyeCalculator´ can compute various quantities: the scattering intensity $I(Q)$ through the Debye scattering equation, the Total Scattering Structure Function $S(Q)$, the Reduced Total Scattering Function $F(Q)$, and the Reduced Atomic Pair Distribution Function $G(r)$. In this section, we specifically outline the $G(r)$ calculation using X-ray scattering. This is because the process for calculating $G(r)$ inherently involves the calculations for $I(Q)$, S(Q), and F(Q). When calling the `gr` function, ´DebyeCalculator´ loads the structure and computes the atomic form factors[@Waasmaier:sh0059]. Following this, it calculates the scattering intensity $I(Q)$ using the Debye scattering equation and subsequently determines the structure factor $S(Q)$ and $F(Q)$. Necessary modifications, such as dampening and Lorch modifications, are applied before computing the $G(r)$. ´DebyeCalculator´ outputs the calculated functions to the CPU by default to allow for immediate analysis of the results, but users have the flexibility to retain the output on the GPU. 
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

In order to benchmark our implementation, we compare simulated scattering patterns from ´DebyeCalculator´ against DiffPy-CMI,[@juhas2015complex] which is a widely recognised software for scattering pattern computations. Here, our implementation obtains the same scattering patterns as DiffPy-CMI (Supporting Information), while being faster on CPU for structures up to ~20,000 atoms (\autoref{fig:figure_1}). Both calculations are run on a x86-64 CPU with 64GB of memory and a batch size of 10,000.
Running the calculations on the GPU provides another notable boost in speed (\autoref{fig:figure_1}). This improvement primarily stems from the distribution of the double sum calculations across a more extensive set of cores than is feasible on the CPU. With smaller atomic structures, an overhead associated with initiating GPU calculations results in the NVIDIA RTX A3000 Laptop GPU computations being slower than DiffPy-CMI and our CPU implementation. Once the atomic structure size exceeds ~14 Å in diameter (~300 atoms), we observe a ~5 times speed-up using an NVIDIA RTX A3000 Laptop GPU with 6GB of memory and a batch size of 10,000. 
The choice of GPU hardware has a substantial influence on this speed advantage. As demonstrated in \autoref{fig:figure_1}, using an NVIDIA Titan RTX GPU, which offers 24GB of memory, the speed benefits become even more evident. The NVIDIA Titan RTX GPU delivers a performance that is ~10 times faster, seemingly across all structure sizes, underlining the significant role of the hardware. With the advancements of GPUs like NVIDIA's Grace Hopper Superchip[@NVIDIA], which boasts 576GB of fast-access to memory, there is potential for ´DebyeCalculator´ to achieve even greater speeds in the future.

![Computation-time comparison of the $G(r)$ calculation using our CPU- and GPU-implementations against DiffPy-CMI[@juhas2015complex]. For the CPU-implementation, a batch size of 10,000 was chosen (x86-64 CPU with 6GB). Both the GPU implementations were run with a batch size of 10,000 (NVIDIA RTX A3000 Laptop GPU with 6GB of memory and NVIDIA Titan RTX GPU with 24GB of memory). The mean and standard deviation of the PDF simulation times are calculated from 10 runs. Note that, due to limited memory, the Laptop GPU was unable to handle structures larger than ~24,000 atoms.\label{fig:figure_1}](../figures/figure_1.png)

# Statement of need

Several software packages already exist for simulating the Debye scattering equation, including DiffPy-CMI[@juhas2015complex], debyer[@debyer], Debussy[@cervellino2010debussy; @cervellino2015debussy], TOPAS[@coelho2018topas], and BCL::SAXS[@putnam2015bcl]. Our software distinguishes itself in several ways. Firstly, it is freely available and open-source licensed under the Apache License 2.0. Moreover, it is conveniently implemented as a ‘pip’ install package and has been integrated with Google Colab[https://github.com/FrederikLizakJohansen/DebyeCalculatorGPU/blob/main/quickstart/QuickStart.ipynb], allowing users to rapidly calculate PD, TS, SAS, and PDF data using the Debye scattering equation without the need of local software installations. ´DebyeCalculator´ can be run through an interactive interface (see \autoref{fig:figure_2}), where users can calculate $I(Q)$, $S(Q)$, $F(Q)$, and $G(r)$ from structural models on both CPU and GPU.

![The interact mode of ´DebyeCalculator´ provides a one-click interface, where the user can update parameters and visualise $I(Q)$, $S(Q)$, $F(Q)$, and $G(r)$. Additionally, the $I(Q)$, $S(Q)$, $F(Q)$, $G(r)$, and xyz-file can be downloaded, including metadata.\label{fig:figure_2}](../figures/figure_2.png)

# Acknowledgements

This work is part of a project that has received funding from the European Research Council (ERC) under the European Union’s Horizon 2020 Research and Innovation Programme (grant agreement No. 804066). We are grateful for funding from University of Copenhagen through the Data+ program.

# Supporting Information

![Comparison of the calculated $I(Q)$, SAXS, $F(Q)$, and $G(r)$ of DebyeCalculator and DiffPy-CMI[@juhas2015complex] on a discrete, spherical cutout with 6 Å in radius from a V~0.985~Al~0.015O2~ crystal[@GHEDIRA1977423].\label{fig:figure_S1}](../figures/figure_S1.png)

# References
