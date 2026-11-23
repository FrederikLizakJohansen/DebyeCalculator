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


# -- compatibility with 1.0.x ----------------------------------------------------------------------------------------

def test_constructor_keeps_parameter_order_of_1_0():
    import inspect
    parameters = list(inspect.signature(DebyeCalculator.__init__).parameters)[1:]
    assert parameters[:18] == ['qmin', 'qmax', 'qstep', 'qdamp', 'rmin', 'rmax', 'rstep', 'rthres', 'biso', 'device',
                               'batch_size', 'lorch_mod', 'radiation_type', 'rad_type', 'profile', '_max_batch_size',
                               '_lightweight_mode', 'dtype']


@pytest.mark.parametrize('batch_size', [None, 1_000])
def test_direct_pair_sum_matches_brute_force_and_grid(batch_size):
    source = particle(8.0)
    direct = DebyeCalculator(device='cpu', dtype=torch.float64, pair_sum='direct', batch_size=batch_size)
    structure = direct._initialize_structures(source)[0]
    pair_iq, _ = direct._compute_iq_parts(structure, include_self_scattering=False)
    assert relative_error(pair_iq.cpu().numpy(), brute_force_pair_iq(direct, source)) < 1e-12

    grid = DebyeCalculator(device='cpu', dtype=torch.float64)
    for name in 'isfg':
        assert relative_error(getattr(direct._get_all(source), name), getattr(grid._get_all(source), name)) < 1e-8


def test_direct_pair_sum_gradients():
    calc = DebyeCalculator(device='cpu', dtype=torch.float64, pair_sum='direct')
    structure = calc._initialize_structures(particle(8.0))[0]
    occupancy = torch.ones(len(structure.elements), dtype=torch.float64)
    grad_xyz, _ = gradients(calc, structure, occupancy, reference=False)
    ref_xyz, _ = gradients(calc, structure, occupancy, reference=True)
    assert (grad_xyz - ref_xyz).norm() / ref_xyz.norm() < 1e-12


def test_batch_size_none_is_automatic():
    assert DebyeCalculator(device='cpu')._effective_batch_size() == 3_000_000
    assert DebyeCalculator(device='cpu', pair_sum='direct')._effective_batch_size() == 10_000
    assert DebyeCalculator(device='cpu', batch_size=50_000)._effective_batch_size() == 50_000
    with pytest.raises(ValueError):
        DebyeCalculator(device='cpu', pair_sum='other')
    with pytest.raises(ValueError):
        DebyeCalculator(device='cpu', num_threads=0)


def test_thread_setting_is_restored_after_overlapping_calls():
    from concurrent.futures import ThreadPoolExecutor
    source = particle(14.0)  # enough pairs for the threaded path
    before = torch.get_num_threads()
    with ThreadPoolExecutor(4) as executor:
        list(executor.map(lambda _: DebyeCalculator(device='cpu').iq(source), range(8)))
    assert torch.get_num_threads() == before


def test_single_thread_leaves_torch_settings_untouched(monkeypatch):
    calls = []
    monkeypatch.setattr(torch, 'set_num_threads', lambda n: calls.append(n))
    DebyeCalculator(device='cpu', num_threads=1).iq(particle(14.0))
    assert calls == []


@pytest.mark.parametrize('version, warns', [('2.14.1', True), ('2.15.0rc1', False), ('2.15.0', False), ('2.16.0.dev20261020', False), ('2.14.1+cpu', True)])
def test_mps_warning_for_pytorch_before_2_15(monkeypatch, version, warns):
    import warnings
    from debyecalculator.debye_calculator import _warn_if_slow_mps_torch
    monkeypatch.setattr(torch, '__version__', version)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _warn_if_slow_mps_torch()
    assert any('PyTorch 2.15' in str(w.message) for w in caught) == warns


def test_mps_warning_only_for_mps(monkeypatch):
    import warnings
    monkeypatch.setattr(torch, '__version__', '2.14.1')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        DebyeCalculator(device='cpu')
    assert not any('PyTorch 2.15' in str(w.message) for w in caught)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='needs an Apple GPU')
def test_mps_warning_from_constructor(monkeypatch):
    monkeypatch.setattr(torch, '__version__', '2.14.1')
    with pytest.warns(UserWarning, match='PyTorch 2.15'):
        DebyeCalculator(device='mps')


# -- native CPU kernels (kernels/) ----------------------------------------------------------------------------------

from debyecalculator.debye_calculator import _NATIVE_BACKENDS, CalculationCancelled

native_backends = pytest.mark.parametrize('backend', sorted(_NATIVE_BACKENDS) or [
    pytest.param(None, marks=pytest.mark.skip(reason='no native pair-sum kernels installed'))])


def with_backend(backend, **kwargs):
    calc = DebyeCalculator(device='cpu', **kwargs)
    calc._pair_backend = backend
    return calc


def pair_iq_and_gradients(calc, structure, occupancy, partial=None):
    xyz = structure.xyz.detach().clone().requires_grad_(True)
    occ = occupancy.to(calc.dtype).clone().requires_grad_(True)
    pair_iq, _ = calc._compute_iq_parts(structure._replace(xyz=xyz, occupancy=occ), partial=partial,
                                        include_self_scattering=False)
    weights = torch.linspace(0.5, 1.5, pair_iq.numel(), dtype=pair_iq.dtype, device=pair_iq.device)
    (pair_iq * weights).sum().backward()
    return pair_iq.detach().cpu().double(), xyz.grad.cpu().double(), occ.grad.cpu().double()


@native_backends
@pytest.mark.parametrize('dtype, tolerance', [(torch.float32, 1e-5), (torch.float64, 1e-12)])
@pytest.mark.parametrize('kwargs, partial, uniform', [
    (dict(), None, True),
    (dict(), None, False),
    (dict(rthres=2.5), None, False),
    (dict(rthres=2.5), None, True),
    (dict(), 'Co-O', False),
    (dict(), 'O-O', True),
    (dict(num_threads=1, batch_size=5_000), None, False),
])
def test_native_kernels_match_torch(backend, dtype, tolerance, kwargs, partial, uniform):
    reference, native = with_backend('torch', dtype=dtype, **kwargs), with_backend(backend, dtype=dtype, **kwargs)
    structure = reference._initialize_structures(particle(10.0))[0]
    assert native._native_pair_kernels(torch.float64, structure.xyz, structure.occupancy) is _NATIVE_BACKENDS[backend]
    assert reference._native_pair_kernels(torch.float64, structure.xyz, structure.occupancy) is None
    n = len(structure.elements)
    occupancy = torch.ones(n, dtype=torch.float64) if uniform else \
        torch.rand(n, dtype=torch.float64, generator=torch.Generator().manual_seed(0)) * 0.5 + 0.5
    for got, want in zip(pair_iq_and_gradients(native, structure, occupancy, partial),
                         pair_iq_and_gradients(reference, structure, occupancy, partial)):
        assert (got - want).norm() / want.norm() < tolerance


@native_backends
def test_native_kernels_threads_and_batches_do_not_change_results(backend):
    structure = DebyeCalculator(device='cpu')._initialize_structures(particle(12.0))[0]
    occupancy = torch.ones(len(structure.elements), dtype=torch.float64)
    reference = pair_iq_and_gradients(with_backend(backend, dtype=torch.float64), structure._replace(
        xyz=structure.xyz.double(), occupancy=structure.occupancy.double()), occupancy)
    calc = with_backend(backend, dtype=torch.float64, num_threads=1, batch_size=1_000)
    got = pair_iq_and_gradients(calc, structure._replace(xyz=structure.xyz.double(),
                                                         occupancy=structure.occupancy.double()), occupancy)
    for g, r in zip(got, reference):
        assert (g - r).norm() / r.norm() < 1e-13


@native_backends
def test_native_kernels_report_progress_and_cancel(backend):
    source = particle(12.0)
    calc = with_backend(backend, num_threads=2, batch_size=20_000)
    fractions = []
    calc.progress_callback = fractions.append
    calc.iq(source)
    assert len(fractions) > 1 and fractions[-1] == pytest.approx(1.0)

    calc.progress_callback = lambda fraction: False
    with pytest.raises(CalculationCancelled):
        calc.iq(source)


@native_backends
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_native_kernels_reject_invalid_input(backend, dtype):
    module = _NATIVE_BACKENDS[backend]
    suffix = 'f32' if dtype == np.float32 else 'f64'
    spread, backward = getattr(module, 'spread_pairs_' + suffix), getattr(module, 'grid_backward_' + suffix)
    xyz = np.array([0, 0, 0, 1, 0, 0, 0, 2, 0], dtype=dtype)
    idx = np.arange(3, dtype=np.int64)
    hist = np.zeros(4 * 1000)

    spread(xyz, None, idx, idx, 0, 3, True, 2.0 ** 8, 0.0, hist)
    assert hist[0::4].sum() == 3  # three pairs i < j
    with pytest.raises(ValueError):  # atom index out of range
        spread(xyz, None, idx, np.array([0, 5], dtype=np.int64), 0, 3, False, 2.0 ** 8, 0.0, hist)
    with pytest.raises(ValueError):  # distance beyond the grid
        spread(xyz, None, idx, idx, 0, 3, True, 2.0 ** 8, 0.0, np.zeros(4 * 10))
    with pytest.raises(ValueError):  # NaN position
        spread(np.where(np.arange(9) == 4, np.nan, xyz).astype(dtype), None, idx, idx, 0, 3, True, 2.0 ** 8, 0.0, hist)
    with pytest.raises(ValueError):  # rows beyond the element
        spread(xyz, None, idx, idx, 0, 4, True, 2.0 ** 8, 0.0, hist)
    with pytest.raises((TypeError, ValueError)):  # wrong dtype
        spread(xyz.astype(np.float16), None, idx, idx, 0, 3, True, 2.0 ** 8, 0.0, hist)

    table, occupancy = np.ones(1003, dtype=dtype), np.ones(3, dtype=dtype)
    grad_xyz, grad_occupancy = np.zeros(9), np.zeros(3)
    backward(xyz, occupancy, idx, idx, 0, 3, True, 2.0 ** 8, 0.0, table, table, grad_xyz, grad_occupancy)
    assert np.allclose(grad_occupancy, 2.0)  # each atom pairs with two others, k = 1
    with pytest.raises(ValueError):  # gradient without its table
        backward(xyz, occupancy, idx, idx, 0, 3, True, 2.0 ** 8, 0.0, None, table, grad_xyz, grad_occupancy)


# -- fused Metal kernels (debyecalculator/metal_pair_sum.py) ---------------------------------------------------------

def _metal_available():
    try:
        from debyecalculator import metal_pair_sum
        return metal_pair_sum.library() is not None
    except Exception:
        return False


metal = pytest.mark.skipif(not _metal_available(), reason='Metal kernels need an Apple GPU and PyTorch 2.6 or later')


def mps_source(radius):
    elements, xyz = particle(radius)
    return elements, xyz.astype(np.float32)


@metal
@pytest.mark.parametrize('kwargs, partial, uniform', [
    (dict(), None, True),
    (dict(), None, False),
    (dict(rthres=2.5), None, False),
    (dict(), 'Co-O', True),
])
def test_metal_kernels_match_float64(kwargs, partial, uniform):
    source = mps_source(10.0)
    reference = DebyeCalculator(device='cpu', dtype=torch.float64, **kwargs)
    reference._pair_backend = 'torch'
    calc = DebyeCalculator(device='mps', dtype=torch.float32, **kwargs)
    calc._pair_backend = 'metal'
    structure_ref = reference._initialize_structures(source)[0]
    structure = calc._initialize_structures(source)[0]
    assert calc._metal_pair_kernels(structure.xyz, structure.occupancy) is not None
    n = len(structure.elements)
    occupancy = torch.ones(n, dtype=torch.float64) if uniform else \
        torch.rand(n, dtype=torch.float64, generator=torch.Generator().manual_seed(0)) * 0.5 + 0.5
    got = pair_iq_and_gradients(calc, structure, occupancy.float().to('mps'), partial)
    want = pair_iq_and_gradients(reference, structure_ref, occupancy, partial)
    for g, w in zip(got, want):
        assert (g.cpu() - w).norm() / w.norm() < 1e-4


@metal
def test_metal_kernels_report_progress_and_cancel():
    source = mps_source(12.0)
    calc = DebyeCalculator(device='mps', batch_size=20_000)
    calc._pair_backend = 'metal'
    fractions = []
    calc.progress_callback = fractions.append
    calc.iq(source)
    assert len(fractions) > 1 and fractions[-1] == pytest.approx(1.0)
    calc.progress_callback = lambda fraction: False
    with pytest.raises(CalculationCancelled):
        calc.iq(source)


@metal
def test_metal_kernels_fall_back_for_negative_occupancy():
    calc = DebyeCalculator(device='mps')
    calc._pair_backend = 'metal'
    xyz = torch.zeros((4, 3), device='mps')
    assert calc._metal_pair_kernels(xyz, torch.ones(4, device='mps')) is not None
    assert calc._metal_pair_kernels(xyz, torch.tensor([1.0, -0.5, 1.0, 1.0], device='mps')) is None
    calc._pair_backend = 'torch'
    assert calc._metal_pair_kernels(xyz, torch.ones(4, device='mps')) is None


# -- NVIDIA GPUs: PyTorch on CUDA and the fused Triton kernels (debyecalculator/triton_pair_sum.py) -----------------

def _triton_available():
    try:
        from debyecalculator import triton_pair_sum
        return triton_pair_sum.library() is not None
    except Exception:
        return False


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs an NVIDIA GPU')
triton_kernels = pytest.mark.skipif(not _triton_available(), reason='Triton kernels need an NVIDIA GPU and Triton')


@cuda
def test_triton_kernels_are_available_on_cuda():
    from debyecalculator import triton_pair_sum
    assert triton_pair_sum.library() is not None, f'Triton kernels unavailable: {triton_pair_sum.unavailable_reason()}'


@cuda
@pytest.mark.parametrize('backend', ['torch', 'triton'])
@pytest.mark.parametrize('dtype, tolerance', [(torch.float32, 1e-5), (torch.float64, 1e-10)])
@pytest.mark.parametrize('kwargs, partial, uniform', [
    (dict(), None, True),
    (dict(), None, False),
    (dict(rthres=2.5), None, False),
    (dict(), 'Co-O', True),
    (dict(), 'O-O', False),
])
def test_cuda_backends_match_float64(backend, dtype, tolerance, kwargs, partial, uniform):
    if backend == 'triton' and not _triton_available():
        pytest.skip('Triton kernels unavailable')
    source = particle(10.0)
    reference = with_backend('torch', dtype=torch.float64, **kwargs)
    calc = DebyeCalculator(device='cuda', dtype=dtype, **kwargs)
    calc._pair_backend = backend
    structure_ref = reference._initialize_structures(source)[0]
    structure = calc._initialize_structures(source)[0]
    expected = backend == 'triton'
    assert (calc._triton_pair_kernels(torch.float64, structure.xyz, structure.occupancy) is not None) == expected
    n = len(structure.elements)
    occupancy = torch.ones(n, dtype=torch.float64) if uniform else \
        torch.rand(n, dtype=torch.float64, generator=torch.Generator().manual_seed(0)) * 0.5 + 0.5
    got = pair_iq_and_gradients(calc, structure, occupancy.to(device='cuda', dtype=dtype), partial)
    want = pair_iq_and_gradients(reference, structure_ref, occupancy, partial)
    for g, w in zip(got, want):
        assert (g - w).norm() / w.norm() < tolerance


@triton_kernels
def test_triton_kernels_match_torch_cuda_on_threaded_sizes():
    source = particle(14.0)
    results = {}
    for backend in ('torch', 'triton'):
        calc = DebyeCalculator(device='cuda', dtype=torch.float64, batch_size=50_000)
        calc._pair_backend = backend
        structure = calc._initialize_structures(source)[0]
        occupancy = torch.ones(len(structure.elements), dtype=torch.float64, device='cuda')
        results[backend] = pair_iq_and_gradients(calc, structure, occupancy)
    for g, w in zip(results['triton'], results['torch']):
        assert (g - w).norm() / w.norm() < 1e-10


@triton_kernels
def test_triton_kernels_report_progress_and_cancel():
    source = particle(12.0)
    calc = DebyeCalculator(device='cuda', batch_size=20_000)
    calc._pair_backend = 'triton'
    fractions = []
    calc.progress_callback = fractions.append
    calc.iq(source)
    assert len(fractions) > 1 and fractions[-1] == pytest.approx(1.0)
    calc.progress_callback = lambda fraction: False
    with pytest.raises(CalculationCancelled):
        calc.iq(source)
