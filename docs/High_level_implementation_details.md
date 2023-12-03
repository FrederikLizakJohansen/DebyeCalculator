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