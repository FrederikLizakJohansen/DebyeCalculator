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


# -- gradients -------------------------------------------------------------------------------------------------------

def exact_pair_iq_torch(calc, structure, xyz, occupancy, partial=None):
    """
    Differentiable brute-force pair_iq in float64 (plain autograd), as reference for the gradients.
    """
    q = calc.q.squeeze(-1).double()
    elements = np.asarray(structure.elements)
    form_factors = structure.unique_form_factors.double()[structure.structure_inverse]
    i, j = torch.triu_indices(len(elements), len(elements), 1)
    if partial is not None:
        a, b = partial.split('-')
        keep = torch.from_numpy(((elements[i] == a) & (elements[j] == b)) | ((elements[i] == b) & (elements[j] == a)))
        i, j = i[keep], j[keep]
    d = (xyz[i] - xyz[j]).norm(dim=1)
    keep = d >= calc.rthres
    i, j, d = i[keep], j[keep], d[keep]
    out = torch.zeros_like(q)
    for chunk in torch.split(torch.arange(d.numel()), 20000):
        weight = (occupancy[i[chunk]] * occupancy[j[chunk]]).unsqueeze(-1) * form_factors[i[chunk]] * form_factors[j[chunk]]
        out = out + (weight * torch.sinc(d[chunk].unsqueeze(-1) * q / np.pi)).sum(0)
    return out * torch.exp(-q ** 2 * calc.biso / (8 * np.pi ** 2))


def gradients(calc, structure, occupancy, reference: bool, partial=None):
    dtype = torch.float64 if reference else calc.dtype
    xyz = structure.xyz.detach().to(dtype).clone().requires_grad_(True)
    occ = occupancy.to(dtype).clone().requires_grad_(True)
    weights = torch.linspace(0.5, 1.5, calc.q.numel(), dtype=torch.float64)
    if reference:
        pair_iq = exact_pair_iq_torch(calc, structure, xyz, occ, partial)
    else:
        pair_iq, _ = calc._compute_iq_parts(structure._replace(xyz=xyz, occupancy=occ), partial=partial,
                                            include_self_scattering=False)
    (pair_iq.double() * weights).sum().backward()
    return xyz.grad.double(), occ.grad.double()


@pytest.mark.parametrize('radius, dtype, tolerance', [
    (3.0, torch.float64, 1e-9),   # direct pair sum (plain autograd)
    (10.0, torch.float64, 1e-9),  # distance grid, custom backward
    (10.0, torch.float32, 1e-5),
    (12.0, torch.float32, 1e-5),  # distance grid with threaded batches
])
def test_gradients_match_exact_autograd(radius, dtype, tolerance):
    calc = DebyeCalculator(device='cpu', dtype=dtype)
    structure = calc._initialize_structures(particle(radius))[0]
    occupancy = torch.rand(len(structure.elements), dtype=torch.float64, generator=torch.Generator().manual_seed(0)) * 0.5 + 0.5
    grad_xyz, grad_occ = gradients(calc, structure, occupancy, reference=False)
    ref_xyz, ref_occ = gradients(calc, structure, occupancy, reference=True)
    assert (grad_xyz - ref_xyz).norm() / ref_xyz.norm() < tolerance
    assert (grad_occ - ref_occ).norm() / ref_occ.norm() < tolerance


@pytest.mark.parametrize('kwargs, partial', [(dict(rthres=2.5), None), (dict(), 'Co-O')])
def test_gradients_with_rthres_and_partial(kwargs, partial):
    calc = DebyeCalculator(device='cpu', dtype=torch.float64, **kwargs)
    structure = calc._initialize_structures(particle(10.0))[0]
    occupancy = torch.ones(len(structure.elements), dtype=torch.float64)
    grad_xyz, _ = gradients(calc, structure, occupancy, reference=False, partial=partial)
    ref_xyz, _ = gradients(calc, structure, occupancy, reference=True, partial=partial)
    assert (grad_xyz - ref_xyz).norm() / ref_xyz.norm() < 1e-9


def test_gradients_through_public_api():
    elements, xyz0 = particle(10.0)
    calc = DebyeCalculator(device='cpu', dtype=torch.float64)
    xyz = torch.tensor(xyz0, requires_grad=True)

    def loss(positions):
        r, gr = calc.gr((elements, positions), keep_on_device=True)
        return (gr * torch.linspace(0, 1, gr.numel(), dtype=gr.dtype)).sum()

    loss(xyz).backward()
    direction = torch.randn(xyz.shape, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    h = 1e-5
    with torch.no_grad():
        finite_difference = (loss(torch.tensor(xyz0) + h * direction) - loss(torch.tensor(xyz0) - h * direction)) / (2 * h)
    assert (xyz.grad * direction).sum().item() == pytest.approx(finite_difference.item(), rel=1e-6)

    # Without keep_on_device, results are numpy arrays also when the positions require gradients
    assert isinstance(calc.iq((elements, xyz))[1], np.ndarray)
