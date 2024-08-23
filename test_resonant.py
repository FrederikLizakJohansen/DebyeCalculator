from debyecalculator import DebyeCalculator
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

calc = DebyeCalculator()

path = "data/AntiFluorite_Co2O.cif"
energies = [7.709, 800]#np.linspace(10,800,2)
print(energies)

fig, axes = plt.subplots(len(energies)+2, 1, figsize=(12,16))

iqs= []
for ax, en in tqdm(zip(axes, energies), total=len(energies)):
    q, iq = calc.iq(path, radii=5, energy=en)
    iqs.append(iq)

    ax.plot(q, iq)

axes[-2].plot(q, iqs[0] - iqs[1], c='g')
q, iq = calc.gr(path, radii=5, energy=None)
axes[-1].plot(q, iq, c='r')

fig.tight_layout()
fig.savefig("dispersive_test.png")

plt.close(fig)

calc = DebyeCalculator()

path = "data/AntiFluorite_Co2O.cif"
energies = [0.1, 800]#np.linspace(10,800,2)
print(energies)

fig, axes = plt.subplots(len(energies)+2, 1, figsize=(12,16))

iqs= []
for ax, en in tqdm(zip(axes, energies), total=len(energies)):
    q, iq = calc.iq(path, radii=5, energy=en)
    iqs.append(iq)

    ax.plot(q, iq)

axes[-2].plot(q, iqs[0] - iqs[1], c='g')
q, iq = calc.gr(path, radii=5, energy=None)
axes[-1].plot(q, iq, c='r')

fig.tight_layout()
fig.savefig("dispersive_test_2.png")

plt.close(fig)
