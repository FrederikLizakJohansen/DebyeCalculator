# The scattering intensity - $I(Q)$

The Debye scattering equation is used to calculate scattering intensities from atomic structures considering the position of each atom in the structure:

$$
I(Q) = \sum_{\nu=1}^{N} \sum_{\mu=1}^{N} b_{\nu}(Q) b_{\mu}(Q) \frac{\sin(Qr_{\nu\mu})}{Qr_{\nu\mu}}
$$

In this equation $Q$ is the momentum transfer of the scattered radiation, $N$ is the number of atoms in the structure, and $r_{\nu\mu}$ is the distance between atoms $\nu$ and $\mu$. For X-ray radiation, the atomic form factor, $b$, depends strongly on $Q$ and is usually denoted as $f(Q)$, but for neutrons, $b$ is independent of $Q$ and referred to as the scattering length. 

# The Total Scattering Structure Function - $S(Q)$

The Total Scattering Structure Functionm, $S(Q)$, is calculated as:

$$
S(Q) = \frac{I_{\text{coh}}(Q) + \langle b(Q) \rangle^2 - \langle b(Q)^2 \rangle}{N \langle b(Q) \rangle^2} - 1
$$

Where $I_{coh}$ is the scattering intensity as we only simulate the coherent scattering signal.

# The Reduced Total Scattering Function - $F(Q)$

The Reduced Total Scattering Function, $F(Q)$, is calculated as:

$$
F(Q) = Q \left( S(Q) \right)
$$

# The Reduced Atomic Pair Distribution Function - $G(r)$

The Reduced Atomic Pair Distribution Function, $G(r)$, is calculated as:

$$
G(r) = \frac{2}{\pi} \int_{Q_{\text{min}}}^{Q_{\text{max}}} F(Q) \sin(Qr) \dQ
$$

Where $Q_{min}$ and $Q_{max}$ is the minimum and maximum Q-values of the data.