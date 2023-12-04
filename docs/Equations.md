## The scattering intensity - $I(Q)$

The Debye scattering equation is used to calculate scattering intensities from atomic structures considering the position of each atom in the structure:

$$
I(Q) = \sum_{\nu=1}^{N} \sum_{\mu=1}^{N} b_{\nu} b_{\mu} \frac{\sin(Qr_{\nu\mu})}{Qr_{\nu\mu}}
$$

In this equation $Q$ is the momentum transfer of the scattered radiation, $N$ is the number of atoms in the structure, and $r_{\nu\mu}$ is the distance between atoms $\nu$ and $\mu$. For X-ray radiation, the atomic form factor, $b$, depends strongly on $Q$ and is usually denoted as $f(Q)$, but for neutrons, $b$ is independent of $Q$ and referred to as the scattering length.

## The Total Scattering Structure Function - $S(Q)$

The Total Scattering Structure Functionm, $S(Q)$, is calculated as:

$$
S(Q) = \frac{1}{N} \frac{I(Q)}{\langle b(Q) \rangle^2}
$$

The normalisation is performed over the compositional average of the form factor (scattering length) contributions, $\langle b \rangle^2 = \sum_µ c_µ b_µ$, where $c_µ$ is the contribution fraction of element $µ$ as well as the number of atoms $N$. Certain conventions will have the total scattering structure function converge to unity at large $Q$. Our implementation is different in that $S(Q)$ will converge to nullity. This however can be rectified by simply adding unity to the above equation.

## The Reduced Total Scattering Function - $F(Q)$

The Reduced Total Scattering Function, $F(Q)$, is calculated as:

$$
F(Q) = Q S(Q)
$$

## The Reduced Atomic Pair Distribution Function - $G(r)$

The inverse Fourier transform of the reduced structure function yields the reduced atomic pair distribution function (PDF), $G(r)$:

$$
G(r) = \frac{2}{\pi} \int_{Q_{\text{min}}}^{Q_{\text{max}}} F(Q) \sin(Qr)dQ
$$

Here the Fourier limits $Q=0$ to $Q=\infty$ is replaced with experimental limits $Q_{\text{min}}$ and $Q_{\text{max}}$, as reaching zero or infinity is not practical. $G(r)$ gives a measure of relative probability of finding two atoms separated by a distance, $r$.

