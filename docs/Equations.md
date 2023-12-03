# I(Q)

The Debye scattering equation is used to calculate scattering intensities from atomic structures considering the position of each atom in the structure:

$$
I(Q) = \sum_{\nu=1}^{N} \sum_{\mu=1}^{N} b_{\nu} b_{\mu} \frac{\sin(Qr_{\nu\mu})}{Qr_{\nu\mu}}
$$

In this equation $Q$ is the momentum transfer of the scattered radiation, $N$ is the number of atoms in the structure, and $r_{\nu\mu}$ is the distance between atoms $\nu$ and $\mu$. For X-ray radiation, the atomic form factor, $b$, depends strongly on $Q$ and is usually denoted as $f(Q)$, but for neutrons, $b$ is independent of $Q$ and referred to as the scattering length. 

# S(Q)

$$
S(Q) = \frac{I_{\text{coh}}(Q) + \langle f(Q) \rangle^2 - \langle f(Q)^2 \rangle}{N \langle f(Q) \rangle^2} - 1
$$

# F(Q)

$$
F(Q) = Q \left( S(Q) \right)
$$


# G(r)

$$
G(r) = \frac{2}{\pi} \int_{Q_{\text{min}}}^{Q_{\text{max}}} F(Q) \sin(Qr) \, dQ
$$
