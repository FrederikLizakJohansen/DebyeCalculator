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
from debyecalculator.debye_calculator import CalculationCancelled
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
    include_self_scattering: bool = True  # in I(Q); S(Q), F(Q) and G(r) use the pair contribution only
    num_threads: int = 0  # CPU threads; 0 keeps the PyTorch default

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
    show_partials: bool = False  # also calculate every element-pair partial
    partials_only: bool = False  # hide the total curve while showing every partial

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
class StructureView:
    """Atoms and optional lattice vectors for the 3D structure viewer."""

    elements: List[str]
    xyz: np.ndarray
    lattice: Optional[np.ndarray] = None
    atom_count: Optional[int] = None


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
    # Element-pair partials {'Co-O': {'i': ..., 's': ..., 'f': ..., 'g': ...}}, summing to the total
    partials: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)


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


def _cell_with_boundary_images(elements: List[str], fractional: np.ndarray,
                               lattice: np.ndarray) -> StructureView:
    """Wrap atoms into a cell and include their images on opposite cell boundaries."""
    fractional = np.mod(np.asarray(fractional, dtype=np.float64), 1.0)
    fractional[np.isclose(fractional, 1.0, atol=1e-7)] = 0.0
    displayed_elements, displayed_fractional, seen = [], [], set()
    for element, coordinate in zip(elements, fractional):
        boundary_axes = np.flatnonzero(np.isclose(coordinate, 0.0, atol=1e-7))
        for mask in range(1 << len(boundary_axes)):
            image = coordinate.copy()
            for bit, axis in enumerate(boundary_axes):
                if mask & (1 << bit):
                    image[axis] = 1.0
            key = (element,) + tuple(np.round(image, 8))
            if key not in seen:
                seen.add(key)
                displayed_elements.append(element)
                displayed_fractional.append(image)
    positions = np.asarray(displayed_fractional, dtype=np.float64) @ lattice
    return StructureView(displayed_elements, positions, lattice, atom_count=len(elements))


def load_unit_cell(path: str, mode: str = 'input') -> StructureView:
    """Load one of the unit-cell representations offered by Materials Project."""
    from ase.io import read

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        atoms = read(path)
    lattice = np.asarray(atoms.cell.array, dtype=np.float64)
    if getattr(atoms.cell, 'rank', np.linalg.matrix_rank(lattice)) < 3:
        raise ValueError('This structure does not contain a three-dimensional unit cell')

    if mode == 'input':
        fractional = np.asarray(atoms.get_scaled_positions(wrap=True), dtype=np.float64)
        return _cell_with_boundary_images(list(atoms.get_chemical_symbols()), fractional, lattice)

    try:
        from pymatgen.io.ase import AseAtomsAdaptor
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    except ImportError as error:
        raise ValueError('This unit-cell mode requires pymatgen (Python 3.10 or newer)') from error

    structure = AseAtomsAdaptor.get_structure(atoms)
    if mode == 'primitive':
        structure = structure.get_primitive_structure()
    elif mode == 'conventional':
        structure = SpacegroupAnalyzer(structure, symprec=0.1).get_conventional_standard_structure()
    elif mode == 'reduced_niggli':
        structure = structure.get_reduced_structure(reduction_algo='niggli')
    elif mode == 'reduced_lll':
        structure = structure.get_reduced_structure(reduction_algo='LLL')
    else:
        raise ValueError(f'Unknown unit-cell mode: {mode}')

    lattice = np.asarray(structure.lattice.matrix, dtype=np.float64)
    elements = [site.specie.symbol for site in structure]
    return _cell_with_boundary_images(elements, np.asarray(structure.frac_coords), lattice)


class Engine:
    def __init__(self, particle_cache_size: int = 64, result_cache_size: int = 64, cell_cache_size: int = 32):
        self._calc: Optional[DebyeCalculator] = None
        self._calc_key = None
        self._applied: Dict[str, object] = {}
        self._particles = LRUCache(particle_cache_size)
        self._q_results = LRUCache(result_cache_size)
        self._cells = LRUCache(cell_cache_size)

    def particle(self, spec: StructureSpec) -> Particle:
        key = spec.particle_key()
        particle = self._particles.get_item(key)
        if particle is None:
            particle = load_particle(spec)
            self._particles.put(key, particle)
        return particle

    def unit_cell(self, path: str, mode: str) -> StructureView:
        key = (str(Path(path).resolve()), mode)
        structure = self._cells.get_item(key)
        if structure is None:
            structure = load_unit_cell(path, mode)
            self._cells.put(key, structure)
        return structure

    def _calculator(self, params: Parameters) -> DebyeCalculator:
        dtype = getattr(torch, params.dtype)
        key = (params.device, params.dtype, params.batch_size, params.num_threads)
        values = {
            'qmin': params.qmin, 'qmax': params.qmax, 'qstep': params.effective_qstep(),
            'rmin': params.rmin, 'rmax': params.rmax, 'rstep': params.rstep,
            'qdamp': params.qdamp, 'biso': params.biso, 'rthres': params.rthres,
            'lorch_mod': params.lorch_mod, 'radiation_type': params.radiation_type,
        }
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if self._calc is None or key != self._calc_key:
                self._calc = DebyeCalculator(device=params.device, dtype=dtype, batch_size=params.batch_size,
                                             num_threads=params.num_threads or None, **values)
                self._calc_key = key
                self._applied = dict(values)
            else:
                changed = {k: v for k, v in values.items() if self._applied.get(k) != v}
                if changed:
                    self._calc.update_parameters(**changed)
                    self._applied.update(changed)
        return self._calc

    def compute(self, params: Parameters, specs: List[StructureSpec], progress=None) -> List[Result]:
        """
        Compute all functions for every structure. Raises on invalid parameters or unreadable files.

        progress: optional callable receiving the completed fraction (0 to 1); returning False cancels the
        calculation with CalculationCancelled.
        """
        if params.qmax <= params.qmin:
            raise ValueError('Qmax must be larger than Qmin')
        if params.rmax <= params.rmin:
            raise ValueError('rmax must be larger than rmin')

        calc = self._calculator(params)
        q_key_params = tuple(getattr(params, name) for name in Q_PARAMETERS) + (
            params.effective_qstep(), params.device, params.dtype)

        particles = [self.particle(spec) for spec in specs]
        tasks = []  # (spec index, partial or None, role): role 'total' or 'breakdown'
        for index, (spec, particle) in enumerate(zip(specs, particles)):
            pairs = particle.element_pairs()
            partial = spec.partial if spec.partial in pairs else None
            tasks.append((index, partial, 'total'))
            if spec.show_partials and partial is None and len(pairs) > 1:
                tasks += [(index, pair, 'breakdown') for pair in pairs]

        totals: Dict[int, tuple] = {}
        breakdowns: Dict[int, Dict[str, Dict[str, np.ndarray]]] = {index: {} for index in range(len(specs))}
        for task_number, (index, partial, role) in enumerate(tasks):
            spec, particle = specs[index], particles[index]
            if progress is not None:
                if progress(task_number / len(tasks)) is False:
                    raise CalculationCancelled()
                calc.progress_callback = lambda fraction, n=task_number: progress((n + fraction) / len(tasks))
            try:
                pair_iq, self_iq, sq, fq = self._q_space(calc, spec, particle, partial, q_key_params)
            finally:
                calc.progress_callback = None
            if role == 'total':
                iq = pair_iq + self_iq if params.include_self_scattering else pair_iq
                totals[index] = (iq, sq, fq, calc.compute_gr(fq))
            else:
                # Self-scattering belongs to the same-element partials, so the partials sum to the total
                a, b = partial.split('-')
                iq = pair_iq + self_iq if (params.include_self_scattering and a == b) else pair_iq
                breakdowns[index][partial] = {name: value.cpu().numpy() for name, value in
                                              zip('isfg', (iq, sq, fq, calc.compute_gr(fq)))}

        q = calc.q.squeeze(-1).cpu().numpy()
        r = calc.r.squeeze(-1).cpu().numpy()
        results = []
        for index, particle in enumerate(particles):
            iq, sq, fq, gr = (value.cpu().numpy() for value in totals[index])
            results.append(Result(q=q, r=r, i=iq, s=sq, f=fq, g=gr, num_atoms=particle.size,
                                  element_pairs=particle.element_pairs(), partials=breakdowns[index]))
        return results

    def _q_space(self, calc: DebyeCalculator, spec: StructureSpec, particle: Particle, partial: Optional[str],
                 q_key_params: tuple) -> tuple:
        key = spec.particle_key() + (partial,) + q_key_params
        cached = self._q_results.get_item(key)
        if cached is None:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                structure = calc._initialize_structures((list(particle.elements), particle.xyz))[0]
                pair_iq, self_iq = calc._compute_iq_parts(structure, partial)
                sq = calc.compute_sq(pair_iq, structure)
                fq = calc.compute_fq(sq)
            cached = (pair_iq, self_iq, sq, fq)
            self._q_results.put(key, cached)
        return cached
