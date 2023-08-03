---
title: 'A GPU-Accelerated Open-Source Python Package for Rapid Calculation of the Debye Scattering Equation: Applications in Small-Angle Scattering, Powder Scattering, and Total Scattering with Pair Distribution Function Analysis'
tags:
  - Python
  - Scattering
  - Debye Scattering Equation
  - GPU-accelerated
  - Nanoparticles
authors:
  - name: Frederik L. Johansen
    orcid: 0000-0002-8049-8624
    equal-contrib: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Andy S. Anker
    orcid: 0000-0002-7403-6642
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
date: 2 August 2023
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Abstract

XXX

# Introduction

The Debye scattering equation, derived in 1915 by P. Debye, is commonly used to calculate the scattering intensities considering the position of each atom in the structure:[@debye:1915; @scardi:2016]

\begin{equation}\label{eq:Debye}
I(Q) = \sum_{i=1}^{N} \sum_{j=1}^{N} f_i(Q) f_j(Q) \frac{\sin(Qr_{ij})}{Qr_{ij}}
\end{equation}

In this equation, Q is the scattering vector, r<sub>ij</sub> is the distance between atom-pair, i and j, and f is the atomic scattering factor. 
The Debye scattering equation can be used to compute the scattering pattern of any atomic structure and is commonly used to study both crystalline and non-crystalline materials with a range of scattering techniques like powder diffraction (PD), total scattering (TS) with pair distribution function (PDF) and small-angle scattering (SAS).[@scardi:2016] Although the Debye scattering equation is extremely versatile, its applicability has been limited by the double sum of the atoms in the structure which makes the equation computationally expensive to calculate. 
With the advancement in computing technology,[@schaller1997moore] new horizons have opened up for applying the Debye scattering equation to larger materials. Modern central processing Units (CPUs), ranging from tenths to hundreds of cores, offer an opportunity to parallelise the computation, significantly enhancing the computational efficiency. This parallel architecture allows for the distribution of the double sum calculations across multiple cores. Graphics processing units (GPUs) further expand computational possibilities, consisting of hundreds or even thousands of smaller, more efficient cores designed for parallel processing.[@garland2008parallel] Unlike traditional CPUs, GPUs are ideally suited for calculations like the Debye scattering equation, where many computations can be performed simultaneously. By leveraging GPU acceleration, researchers can achieve computational speeds that are orders of magnitude faster than even the most advanced multi-core CPUs.
We introduce a GPU-accelerated open-source Python package for rapid calculation of the scattering intensity from a xyz-file using the Debye scattering equation. The xyz-file format describes the atomic structure with the atomic identify and its xyz-coordinates and is commonly used in materials chemistry. We further calculate the PDF as described in Underneath the Bragg Peaks.[@egami2003underneath] We show that our software can simulate the PD, TS, SAS and PDF data orders of magnitudes faster than on the CPU, while being open-source and easy assessable by other scientists.

# Results & Discussion:

Table: Pseudo-code incl. profiling (times)

![Q and r-space comparison of our software and DiffPy-CMI on two systems - monoatomic (metal) and diatomic (metal oxide): (evt. Topas??).\label{fig:figure1}](../figures/figure1.png){ width=100% }
and referenced from text using \autoref{fig:figure1}.

![CPU vs. GPU (in Q and in r-space) (+ batching.\label{fig:figure2}](../figures/figure2.png)
and referenced from text using \autoref{fig:figure2}.

![GPU time vs. size and #atoms,\label{fig:figure3}](../figures/figure3.png)
and referenced from text using \autoref{fig:figure3}.


# Conclusions

XXX

# Acknowledgements

XXX

# References


