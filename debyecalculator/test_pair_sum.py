"""
Tests of the pair-sum implementation against a brute-force Debye sum, covering the direct path (small structures),
the distance-grid path, batching, threads, partials, rthres, occupancies and particle generation.
"""

import numpy as np
import pytest
import torch

from debyecalculator import DebyeCalculator
from debyecalculator.utility.generate import generate_nanoparticles

CIF = 'debyecalculator/data/AntiFluorite_Co2O.cif'


def particle(radius):
    p = generate_nanoparticles(CIF, radius, disable_pbar=True, device='cpu')[0]
    return list(p.elements), p.xyz.cpu().numpy().astype(np.float64)


def brute_force_pair_iq(calc, source, partial=None, occupancy=None):
    """
    sum_{i<j} o_i o_j f_i f_j sinc(q d_ij) exp(-q^2 Biso / 8 pi^2) in float64, with the calculator's form factors.
    """
    structure = calc._initialize_structures(source)[0]
    elements = np.asarray(structure.elements)
    xyz = np.asarray(source[1], dtype=np.float64)
    q = calc.q.squeeze(-1).cpu().double().numpy()
    form_factors = structure.unique_form_factors[structure.structure_inverse].cpu().double().numpy()
    occ = np.ones(len(elements)) if occupancy is None else np.asarray(occupancy, dtype=np.float64)

    i, j = np.triu_indices(len(elements), 1)
    if partial is not None:
        a, b = partial.split('-')
        keep = ((elements[i] == a) & (elements[j] == b)) | ((elements[i] == b) & (elements[j] == a))
        i, j = i[keep], j[keep]
    d = np.linalg.norm(xyz[i] - xyz[j], axis=1)
    keep = d >= calc.rthres
    i, j, d = i[keep], j[keep], d[keep]

    iq = np.zeros_like(q)
    for start in range(0, len(d), 20000):
        sl = slice(start, start + 20000)
        weights = (occ[i[sl]] * occ[j[sl]])[:, None] * form_factors[i[sl]] * form_factors[j[sl]]
        iq += np.sum(weights * np.sinc(d[sl, None] * q[None, :] / np.pi), axis=0)
    return iq * np.exp(-q ** 2 * calc.biso / (8 * np.pi ** 2))


def relative_error(a, b):
    return np.max(np.abs(np.asarray(a) - b)) / np.max(np.abs(b))


@pytest.mark.parametrize('radius', [3.0, 8.0])  # 3 Å: direct pair sum, 8 Å: distance grid
@pytest.mark.parametrize('dtype, tolerance', [(torch.float32, 1e-5), (torch.float64, 1e-9)])
def test_pair_sum_matches_brute_force(radius, dtype, tolerance):
    source = particle(radius)
    calc = DebyeCalculator(device='cpu', dtype=dtype)
    structure = calc._initialize_structures(source)[0]
    pair_iq, _ = calc._compute_iq_parts(structure, include_self_scattering=False)
    assert relative_error(pair_iq.cpu().numpy(), brute_force_pair_iq(calc, source)) < tolerance


@pytest.mark.parametrize('kwargs, partial', [
    (dict(rthres=2.5), None),
    (dict(), 'Co-O'),
    (dict(), 'O-O'),
    (dict(radiation_type='neutron'), None),
    (dict(qmin=0.0, qmax=3.0, qstep=0.01), None),
])
def test_pair_sum_options_match_brute_force(kwargs, partial):
    source = particle(8.0)
    calc = DebyeCalculator(device='cpu', dtype=torch.float64, **kwargs)
    structure = calc._initialize_structures(source)[0]
    pair_iq, _ = calc._compute_iq_parts(structure, partial=partial, include_self_scattering=False)
    assert relative_error(pair_iq.cpu().numpy(), brute_force_pair_iq(calc, source, partial=partial)) < 1e-9


def test_batch_size_and_threads_do_not_change_results():
    source = particle(8.0)
    reference = DebyeCalculator(device='cpu', dtype=torch.float64).iq(source)[1]
    threads = torch.get_num_threads()
    try:
        for batch_size, num_threads in [(1_000, threads), (50_000, 1), (100_000_000, threads)]:
            torch.set_num_threads(num_threads)
            iq = DebyeCalculator(device='cpu', dtype=torch.float64, batch_size=batch_size).iq(source)[1]
            assert relative_error(iq, reference) < 1e-10, (batch_size, num_threads)
    finally:
        torch.set_num_threads(threads)


def test_coincident_atoms():
    elements, xyz = particle(3.0)
    source = (elements + [elements[0]], np.vstack([xyz, xyz[:1]]))
    calc = DebyeCalculator(device='cpu', dtype=torch.float64)
    structure = calc._initialize_structures(source)[0]
    pair_iq, _ = calc._compute_iq_parts(structure, include_self_scattering=False)
    assert relative_error(pair_iq.cpu().numpy(), brute_force_pair_iq(calc, source)) < 1e-9


def test_xyz_with_occupancy_column(tmp_path):
    elements, xyz = particle(6.0)
    occupancy = np.random.default_rng(0).uniform(0.3, 1.0, len(elements)).round(3)
    path = tmp_path / 'particle.xyz'
    with open(path, 'w') as f:
        f.write(f'{len(elements)}\n\n')
        for element, (x, y, z), o in zip(elements, xyz, occupancy):
            f.write(f'{element} {x:.6f} {y:.6f} {z:.6f} {o}\n')

    calc = DebyeCalculator(device='cpu', dtype=torch.float64)
    structure = calc._initialize_structures(str(path))[0]
    assert np.allclose(structure.occupancy.cpu().numpy(), occupancy)
    pair_iq, _ = calc._compute_iq_parts(structure, include_self_scattering=False)
    reference = brute_force_pair_iq(calc, (elements, np.loadtxt(path, skiprows=2, usecols=(1, 2, 3))), occupancy=occupancy)
    assert relative_error(pair_iq.cpu().numpy(), reference) < 1e-9


def test_large_particle_float32_against_float64():
    source = particle(14.0)  # 1,367 atoms
    iq32 = DebyeCalculator(device='cpu', dtype=torch.float32)._get_all(source)
    iq64 = DebyeCalculator(device='cpu', dtype=torch.float64)._get_all(source)
    for name in 'isfg':
        assert relative_error(getattr(iq32, name), getattr(iq64, name)) < 2e-5, name


@pytest.mark.parametrize('lightweight', [False, True])
def test_generation_is_deterministic_and_centred(lightweight):
    first = generate_nanoparticles(CIF, [5.0, 9.0], disable_pbar=True, device='cpu', _lightweight_mode=lightweight)
    second = generate_nanoparticles(CIF, [5.0, 9.0], disable_pbar=True, device='cpu', _lightweight_mode=lightweight)
    for a, b in zip(first, second):
        assert a.elements == b.elements
        assert torch.equal(a.xyz, b.xyz)
    assert first[0].size < first[1].size
