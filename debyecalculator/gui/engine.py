"""
Computation layer of the desktop GUI, independent of Qt.

The engine turns a parameter set and a list of structure specifications into I(Q), S(Q), F(Q) and G(r) for
every structure. It caches generated nanoparticles per (file, radius) and the Q-space results per structure and
Q-space parameters, so that changes to parameters that only enter G(r) (r-range, qdamp, Lorch modification)
reuse the Q-space results.
"""

import math
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from debyecalculator import DebyeCalculator
from debyecalculator.utility.generate import generate_nanoparticles

STRUCTURE_SUFFIXES = ('.cif', '.xyz', '.pdb', '.vasp', '.poscar', '.extxyz', '.traj')

# Parameters that change the Q-space results; all other parameters only enter G(r)
Q_PARAMETERS = ('qmin', 'qmax', 'qstep', 'biso', 'rthres', 'radiation_type')
R_PARAMETERS = ('rmin', 'rmax', 'rstep', 'qdamp', 'lorch_mod')


@dataclass
class Parameters:
    qmin: float = 1.0
    qmax: float = 30.0
    qstep: Optional[float] = None  # None: pi / (rmax + rstep)
    rmin: float = 0.0
    rmax: float = 20.0
    rstep: float = 0.01
    qdamp: float = 0.04
    biso: float = 0.3
    rthres: float = 0.0
    lorch_mod: bool = False
    radiation_type: str = 'xray'
    device: str = 'cpu'
    dtype: str = 'float32'
    batch_size: int = 3_000_000

    def effective_qstep(self) -> float:
        return self.qstep if self.qstep else math.pi / (self.rmax + self.rstep)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StructureSpec:
    path: str
    radius: float = 10.0
    lightweight: bool = False
    partial: Optional[str] = None

    @property
    def is_cif(self) -> bool:
        return Path(self.path).suffix.lower() == '.cif'

    def particle_key(self) -> tuple:
        if self.is_cif:
            return (self.path, round(float(self.radius), 6), self.lightweight)
        return (self.path,)


@dataclass
class Particle:
    elements: List[str]
    xyz: np.ndarray

    @property
    def size(self) -> int:
        return len(self.elements)

    def element_pairs(self) -> List[str]:
        unique = sorted(set(self.elements))
        return [f'{a}-{b}' for i, a in enumerate(unique) for b in unique[i:]]


@dataclass
class Result:
    q: np.ndarray
    r: np.ndarray
    i: np.ndarray
    s: np.ndarray
    f: np.ndarray
    g: np.ndarray
    num_atoms: int
    element_pairs: List[str] = field(default_factory=list)


class LRUCache(OrderedDict):
    def __init__(self, maxsize: int):
        super().__init__()
        self.maxsize = maxsize

    def get_item(self, key):
        if key in self:
            self.move_to_end(key)
            return self[key]
        return None

    def put(self, key, value):
        self[key] = value
        self.move_to_end(key)
        while len(self) > self.maxsize:
            self.popitem(last=False)


def available_devices() -> List[str]:
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        devices.append('mps')
    return devices


def load_particle(spec: StructureSpec) -> Particle:
    """
    Generate a nanoparticle from a CIF, or read a discrete structure from any file format ASE reads.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if spec.is_cif:
            particles = generate_nanoparticles(spec.path, float(spec.radius), disable_pbar=True, device='cpu',
                                               _lightweight_mode=spec.lightweight)
            if not particles:
                raise ValueError(f'No particle with more than one atom at radius {spec.radius} Å')
            particle = particles[0]
            return Particle(list(particle.elements), particle.xyz.cpu().numpy().astype(np.float64))

        from ase.io import read
        atoms = read(spec.path)
        return Particle(list(atoms.get_chemical_symbols()), np.asarray(atoms.get_positions(), dtype=np.float64))


class Engine:
    def __init__(self, particle_cache_size: int = 64, result_cache_size: int = 64):
        self._calc: Optional[DebyeCalculator] = None
        self._calc_key = None
        self._applied: Dict[str, object] = {}
        self._particles = LRUCache(particle_cache_size)
        self._q_results = LRUCache(result_cache_size)

    def particle(self, spec: StructureSpec) -> Particle:
        key = spec.particle_key()
        particle = self._particles.get_item(key)
        if particle is None:
            particle = load_particle(spec)
            self._particles.put(key, particle)
        return particle

    def _calculator(self, params: Parameters) -> DebyeCalculator:
        dtype = getattr(torch, params.dtype)
        key = (params.device, params.dtype, params.batch_size)
        values = {
            'qmin': params.qmin, 'qmax': params.qmax, 'qstep': params.effective_qstep(),
            'rmin': params.rmin, 'rmax': params.rmax, 'rstep': params.rstep,
            'qdamp': params.qdamp, 'biso': params.biso, 'rthres': params.rthres,
            'lorch_mod': params.lorch_mod, 'radiation_type': params.radiation_type,
        }
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if self._calc is None or key != self._calc_key:
                self._calc = DebyeCalculator(device=params.device, dtype=dtype, batch_size=params.batch_size, **values)
                self._calc_key = key
                self._applied = dict(values)
            else:
                changed = {k: v for k, v in values.items() if self._applied.get(k) != v}
                if changed:
                    self._calc.update_parameters(**changed)
                    self._applied.update(changed)
        return self._calc

    def compute(self, params: Parameters, specs: List[StructureSpec]) -> List[Result]:
        """
        Compute all functions for every structure. Raises on invalid parameters or unreadable files.
        """
        if params.qmax <= params.qmin:
            raise ValueError('Qmax must be larger than Qmin')
        if params.rmax <= params.rmin:
            raise ValueError('rmax must be larger than rmin')

        calc = self._calculator(params)
        q_key_params = tuple(getattr(params, name) for name in Q_PARAMETERS) + (
            params.effective_qstep(), params.device, params.dtype)

        results = []
        for spec in specs:
            particle = self.particle(spec)
            partial = spec.partial if spec.partial and spec.partial in particle.element_pairs() else None
            key = spec.particle_key() + (partial,) + q_key_params
            cached = self._q_results.get_item(key)
            if cached is None:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    structure = calc._initialize_structures((list(particle.elements), particle.xyz))[0]
                    pair_iq, self_iq = calc._compute_iq_parts(structure, partial)
                    iq = pair_iq + self_iq
                    sq = calc.compute_sq(pair_iq, structure)
                    fq = calc.compute_fq(sq)
                cached = (iq, sq, fq)
                self._q_results.put(key, cached)
            iq, sq, fq = cached
            gr = calc.compute_gr(fq)
            results.append(Result(
                q=calc.q.squeeze(-1).cpu().numpy(),
                r=calc.r.squeeze(-1).cpu().numpy(),
                i=iq.cpu().numpy(), s=sq.cpu().numpy(), f=fq.cpu().numpy(), g=gr.cpu().numpy(),
                num_atoms=particle.size,
                element_pairs=particle.element_pairs(),
            ))
        return results
