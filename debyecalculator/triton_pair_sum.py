"""
Fused Triton kernels for the grid pair sum on NVIDIA GPUs (device='cuda', float32 and float64). Triton ships with the
Linux CUDA wheels of PyTorch and compiles the kernels at run time; no CUDA toolkit or build step is needed.

The kernels do the per-pair work of DebyeCalculator._spread_pairs (forward) and
DebyeCalculator._grid_pair_sums_backward (backward) in one pass over tiles of BLOCK_R x BLOCK_C atom pairs, without
batch-sized temporaries:

- forward: each program computes the distances of its tile, and adds the bin moments w, w s, w s^2, w s^3 of the valid
  pairs to the float64 histogram with float64 atomic adds (native on compute capability 6.0 and later).
- backward: each program sums the gradient terms of its row atoms over the ordered pairs (i, j) of its tile, and adds
  them to float64 gradients with one atomic add per row atom and tile: no per-pair atomics.

The per-pair arithmetic is in the calculation dtype, with correctly rounded square roots and divisions (sqrt_rn,
div_rn) in float32. On first use, library() runs a small self-test against a float64 reference and reports the kernels
as unavailable (so that the PyTorch code is used) if Triton is missing, fails to compile, or gives wrong results.
"""
import math
import os
import sys

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - depends on the installation
    triton = None
    tl = None

# Tile sizes (pairs per program = BLOCK_R * BLOCK_C) and pairs per forward launch; overridable for tuning
BLOCK_R = int(os.environ.get('DEBYECALCULATOR_TRITON_BLOCK_R', 32))
BLOCK_C = int(os.environ.get('DEBYECALCULATOR_TRITON_BLOCK_C', 64))
NUM_WARPS = int(os.environ.get('DEBYECALCULATOR_TRITON_NUM_WARPS', 4))
PAIRS_PER_LAUNCH = 1 << 31

if triton is not None:

    @triton.jit
    def _sqrt(x, IS_FP32: tl.constexpr):
        if IS_FP32:
            return tl.sqrt_rn(x)
        else:
            return tl.sqrt(x)  # float64 square roots are correctly rounded

    @triton.jit
    def _div(a, b, IS_FP32: tl.constexpr):
        if IS_FP32:
            return tl.div_rn(a, b)
        else:
            return a / b  # float64 division is correctly rounded

    @triton.jit
    def _spread_kernel(
        xyz_ptr, occ_ptr, idx_a_ptr, idx_b_ptr, hist_ptr, error_ptr, rthres_ptr,
        row_start, n_rows, n_b, inv_dr, n_bins,
        SAME: tl.constexpr, WEIGHTED: tl.constexpr, THRESHOLD: tl.constexpr, IS_FP32: tl.constexpr,
        BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr,
    ):
        r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
        c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
        row = row_start + r
        row_ok = r < n_rows
        col_ok = c < n_b
        i = tl.load(idx_a_ptr + row, mask=row_ok, other=0)
        j = tl.load(idx_b_ptr + c, mask=col_ok, other=0)
        xi = tl.load(xyz_ptr + 3 * i, mask=row_ok, other=0.0)
        yi = tl.load(xyz_ptr + 3 * i + 1, mask=row_ok, other=0.0)
        zi = tl.load(xyz_ptr + 3 * i + 2, mask=row_ok, other=0.0)
        xj = tl.load(xyz_ptr + 3 * j, mask=col_ok, other=0.0)
        yj = tl.load(xyz_ptr + 3 * j + 1, mask=col_ok, other=0.0)
        zj = tl.load(xyz_ptr + 3 * j + 2, mask=col_ok, other=0.0)

        valid = row_ok[:, None] & col_ok[None, :]
        if SAME:
            # Within one element only pairs j > i (indices into the same atom list)
            valid = valid & (c[None, :] > row[:, None])
        dx = xi[:, None] - xj[None, :]
        dy = yi[:, None] - yj[None, :]
        dz = zi[:, None] - zj[None, :]
        d = _sqrt(dx * dx + dy * dy + dz * dz, IS_FP32)
        if THRESHOLD:
            # rthres in the calculation dtype (a scalar argument would be float32)
            valid = valid & (d >= tl.load(rthres_ptr))

        u = d * inv_dr
        b = tl.floor(u)
        inside = (b >= 0.0) & (b < n_bins)  # false for NaN
        bad = valid & ((b < 0.0) | (b >= n_bins) | (b != b))
        tl.atomic_add(error_ptr, tl.sum(tl.sum(bad.to(tl.int32), axis=1), axis=0))
        valid = valid & inside

        k = tl.where(valid, b, 0.0).to(tl.int64)
        s = u - b
        s2 = s * s
        s3 = s2 * s
        if WEIGHTED:
            oi = tl.load(occ_ptr + i, mask=row_ok, other=0.0)
            oj = tl.load(occ_ptr + j, mask=col_ok, other=0.0)
            w = oi[:, None] * oj[None, :]
            m0 = w.to(tl.float64)
            m1 = (s * w).to(tl.float64)
            m2 = (s2 * w).to(tl.float64)
            m3 = (s3 * w).to(tl.float64)
        else:
            m0 = tl.full([BLOCK_R, BLOCK_C], 1.0, tl.float64)
            m1 = s.to(tl.float64)
            m2 = s2.to(tl.float64)
            m3 = s3.to(tl.float64)
        h = hist_ptr + 4 * k
        tl.atomic_add(h, m0, mask=valid, sem='relaxed')
        tl.atomic_add(h + 1, m1, mask=valid, sem='relaxed')
        tl.atomic_add(h + 2, m2, mask=valid, sem='relaxed')
        tl.atomic_add(h + 3, m3, mask=valid, sem='relaxed')

    @triton.jit
    def _gradient_kernel(
        xyz_ptr, occ_ptr, idx_a_ptr, idx_b_ptr, h_ptr, k_ptr, grad_xyz_ptr, grad_occ_ptr, error_ptr, rthres_ptr,
        n_rows, n_b, inv_dr, n_bins,
        SKIP_SELF: tl.constexpr, NEED_XYZ: tl.constexpr, NEED_OCC: tl.constexpr, THRESHOLD: tl.constexpr,
        IS_FP32: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr,
    ):
        r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
        c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
        row_ok = r < n_rows
        col_ok = c < n_b
        i = tl.load(idx_a_ptr + r, mask=row_ok, other=0)
        j = tl.load(idx_b_ptr + c, mask=col_ok, other=0)
        xi = tl.load(xyz_ptr + 3 * i, mask=row_ok, other=0.0)
        yi = tl.load(xyz_ptr + 3 * i + 1, mask=row_ok, other=0.0)
        zi = tl.load(xyz_ptr + 3 * i + 2, mask=row_ok, other=0.0)
        xj = tl.load(xyz_ptr + 3 * j, mask=col_ok, other=0.0)
        yj = tl.load(xyz_ptr + 3 * j + 1, mask=col_ok, other=0.0)
        zj = tl.load(xyz_ptr + 3 * j + 2, mask=col_ok, other=0.0)
        oi = tl.load(occ_ptr + i, mask=row_ok, other=0.0)
        oj = tl.load(occ_ptr + j, mask=col_ok, other=0.0)

        valid = row_ok[:, None] & col_ok[None, :]
        if SKIP_SELF:
            valid = valid & (i[:, None] != j[None, :])
        dx = xi[:, None] - xj[None, :]
        dy = yi[:, None] - yj[None, :]
        dz = zi[:, None] - zj[None, :]
        d = _sqrt(dx * dx + dy * dy + dz * dz, IS_FP32)
        if THRESHOLD:
            # rthres in the calculation dtype (a scalar argument would be float32)
            valid = valid & (d >= tl.load(rthres_ptr))

        u = d * inv_dr
        b = tl.floor(u)
        inside = (b >= 0.0) & (b < n_bins)  # false for NaN
        bad = valid & ((b < 0.0) | (b >= n_bins) | (b != b))
        tl.atomic_add(error_ptr, tl.sum(tl.sum(bad.to(tl.int32), axis=1), axis=0))
        valid = valid & inside

        k = tl.where(valid, b, 0.0).to(tl.int64)
        s = u - b
        sp1 = s + 1.0
        sm1 = s - 1.0
        sm2 = s - 2.0
        # Cubic Lagrange weights of nodes k..k+3, in the operation order of the PyTorch code
        six = tl.full([BLOCK_R, BLOCK_C], 6.0, s.dtype)
        two = tl.full([BLOCK_R, BLOCK_C], 2.0, s.dtype)
        w0 = _div(s * sm1 * sm2, -six, IS_FP32)
        w1 = _div(sp1 * sm1 * sm2, two, IS_FP32)
        w2 = _div(sp1 * s * sm2, -two, IS_FP32)
        w3 = _div(sp1 * s * sm1, six, IS_FP32)

        if NEED_XYZ:
            h0 = tl.load(h_ptr + k, mask=valid, other=0.0)
            h1 = tl.load(h_ptr + k + 1, mask=valid, other=0.0)
            h2 = tl.load(h_ptr + k + 2, mask=valid, other=0.0)
            h3 = tl.load(h_ptr + k + 3, mask=valid, other=0.0)
            hv = w0 * h0 + w1 * h1 + w2 * h2 + w3 * h3
            positive = valid & (d > 0.0)
            coefficient = _div(hv, tl.where(positive, d, 1.0), IS_FP32) * (oi[:, None] * oj[None, :])
            coefficient = tl.where(positive, coefficient, 0.0)
            gx = tl.sum(coefficient * dx, axis=1).to(tl.float64)
            gy = tl.sum(coefficient * dy, axis=1).to(tl.float64)
            gz = tl.sum(coefficient * dz, axis=1).to(tl.float64)
            tl.atomic_add(grad_xyz_ptr + 3 * i, gx, mask=row_ok, sem='relaxed')
            tl.atomic_add(grad_xyz_ptr + 3 * i + 1, gy, mask=row_ok, sem='relaxed')
            tl.atomic_add(grad_xyz_ptr + 3 * i + 2, gz, mask=row_ok, sem='relaxed')
        if NEED_OCC:
            k0 = tl.load(k_ptr + k, mask=valid, other=0.0)
            k1 = tl.load(k_ptr + k + 1, mask=valid, other=0.0)
            k2 = tl.load(k_ptr + k + 2, mask=valid, other=0.0)
            k3 = tl.load(k_ptr + k + 3, mask=valid, other=0.0)
            kv = w0 * k0 + w1 * k1 + w2 * k2 + w3 * k3
            go = tl.sum(tl.where(valid, kv * oj[None, :], 0.0), axis=1).to(tl.float64)
            tl.atomic_add(grad_occ_ptr + i, go, mask=row_ok, sem='relaxed')


_state = {'checked': False, 'error': None}


def _check(error):
    if int(error.item()):
        raise ValueError('pair distance outside the distance grid')


def _launch_spread(xyz, occupancy, idx_a, idx_b, hist_row, error, rthres, threshold, row_start, n_rows, same, weighted,
                   inv_dr, n_bins):
    n_b = idx_b.numel()
    grid = (triton.cdiv(n_rows, BLOCK_R), triton.cdiv(n_b, BLOCK_C))
    _spread_kernel[grid](
        xyz, occupancy, idx_a, idx_b, hist_row, error, rthres,
        row_start, n_rows, n_b, inv_dr, float(n_bins),
        SAME=bool(same), WEIGHTED=bool(weighted), THRESHOLD=bool(threshold), IS_FP32=xyz.dtype == torch.float32,
        BLOCK_R=BLOCK_R, BLOCK_C=BLOCK_C, num_warps=NUM_WARPS,
    )


def spread_pairs(jobs, xyz, occupancy, atom_indices, n_element_pairs, n_nodes, dr, rthres, progress=None):
    """
    Bin moments of the given batches, as DebyeCalculator._spread_pairs. Returns a float64 tensor on the device of shape
    (n_element_pairs, n_nodes, 4). Consecutive batches of one element pair run in one launch (up to PAIRS_PER_LAUNCH
    pairs); progress is reported per batch after each launch.
    """
    device = xyz.device
    xyz = xyz.detach().contiguous().reshape(-1)
    weighted = not bool(torch.all(occupancy == 1))
    occupancy = occupancy.detach().contiguous()
    indices = [idx.to(device=device, dtype=torch.int64).contiguous() for idx in atom_indices]
    hist = torch.zeros((n_element_pairs, n_nodes * 4), device=device, dtype=torch.float64)
    error = torch.zeros(1, device=device, dtype=torch.int32)
    rthres_tensor = torch.tensor([float(rthres)], device=device, dtype=xyz.dtype)

    group = []

    def flush():
        if not group:
            return
        pair_number, a, b, start, _ = group[0]
        stop = group[-1][4]
        _launch_spread(xyz, occupancy, indices[a], indices[b], hist[pair_number], error, rthres_tensor,
                       float(rthres) > 0, start, stop - start, a == b, weighted, 1.0 / dr, n_nodes)
        if progress is not None:
            for _ in group:
                progress.step()
        group.clear()

    pairs_in_group = 0
    for job in jobs:
        pair_number, a, b, start, stop = job
        contiguous = group and group[-1][0] == pair_number and group[-1][4] == start
        if group and (not contiguous or pairs_in_group >= PAIRS_PER_LAUNCH):
            flush()
            pairs_in_group = 0
        group.append(job)
        pairs_in_group += (stop - start) * indices[b].numel()
    flush()

    _check(error)
    return hist.view(n_element_pairs, n_nodes, 4)


def pair_gradients(xyz, occupancy, atom_indices, element_pairs, h_tables, k_tables, dr, rthres, need_xyz,
                   need_occupancy):
    """
    Gradients with respect to positions and occupancies, as DebyeCalculator._grid_pair_sums_backward, from the node
    tables (n_element_pairs, n_nodes + 3) in the calculation dtype. Returns float64 tensors on the device (None if not
    needed).
    """
    device = xyz.device
    xyz = xyz.detach().contiguous().reshape(-1)
    occupancy = occupancy.detach().to(xyz.dtype).contiguous()
    indices = [idx.to(device=device, dtype=torch.int64).contiguous() for idx in atom_indices]
    grad_xyz = torch.zeros(xyz.numel(), device=device, dtype=torch.float64)
    grad_occupancy = torch.zeros(occupancy.numel(), device=device, dtype=torch.float64)
    error = torch.zeros(1, device=device, dtype=torch.int32)
    rthres_tensor = torch.tensor([float(rthres)], device=device, dtype=xyz.dtype)
    dummy = torch.zeros(4, device=device, dtype=xyz.dtype)

    for pair_number, (a, b) in enumerate(element_pairs):
        h = h_tables[pair_number].to(xyz.dtype).contiguous() if need_xyz else dummy
        k = k_tables[pair_number].to(xyz.dtype).contiguous() if need_occupancy else dummy
        n_bins = (h if need_xyz else k).numel() - 3
        # Ordered pairs: rows of a against columns of b and, for two elements, rows of b against columns of a
        for rows, cols in ([(a, b)] if a == b else [(a, b), (b, a)]):
            n_rows, n_b = indices[rows].numel(), indices[cols].numel()
            if n_rows == 0 or n_b == 0:
                continue
            grid = (triton.cdiv(n_rows, BLOCK_R), triton.cdiv(n_b, BLOCK_C))
            _gradient_kernel[grid](
                xyz, occupancy, indices[rows], indices[cols], h, k, grad_xyz, grad_occupancy, error, rthres_tensor,
                n_rows, n_b, 1.0 / dr, float(n_bins),
                SKIP_SELF=a == b, NEED_XYZ=bool(need_xyz), NEED_OCC=bool(need_occupancy), THRESHOLD=float(rthres) > 0,
                IS_FP32=xyz.dtype == torch.float32, BLOCK_R=BLOCK_R, BLOCK_C=BLOCK_C, num_warps=NUM_WARPS,
            )
    _check(error)
    return (grad_xyz if need_xyz else None), (grad_occupancy if need_occupancy else None)


def _self_test():
    """Bin moments and gradients of a few random atoms against a float64 reference computed with PyTorch."""
    device = torch.device('cuda')
    generator = torch.Generator().manual_seed(0)
    n, dr = 40, 2.0 ** -6
    xyz64 = torch.rand((n, 3), generator=generator, dtype=torch.float64) * 4
    occupancy64 = torch.rand(n, generator=generator, dtype=torch.float64) * 0.5 + 0.5
    split = n // 2
    atom_indices = [torch.arange(split), torch.arange(split, n)]
    element_pairs = [(0, 0), (0, 1), (1, 1)]
    n_nodes = int(2 * 4 * math.sqrt(3) / dr) + 8
    jobs = [(t, a, b, 0, atom_indices[a].numel()) for t, (a, b) in enumerate(element_pairs)]
    h_tables = torch.rand((3, n_nodes + 3), generator=generator, dtype=torch.float64)
    k_tables = torch.rand((3, n_nodes + 3), generator=generator, dtype=torch.float64)

    # Reference: every pair of each element pair once (i < j within an element)
    hist_ref = torch.zeros((3, n_nodes, 4), dtype=torch.float64)
    grad_xyz_ref = torch.zeros((n, 3), dtype=torch.float64)
    grad_occ_ref = torch.zeros(n, dtype=torch.float64)
    for t, (a, b) in enumerate(element_pairs):
        for p, i in enumerate(atom_indices[a].tolist()):
            for q, j in enumerate(atom_indices[b].tolist()):
                if a == b and q <= p:
                    continue
                diff = xyz64[i] - xyz64[j]
                d = diff.norm()
                u = d / dr
                kk = int(math.floor(u))
                s = float(u - kk)
                w = occupancy64[i] * occupancy64[j]
                hist_ref[t, kk] += w * torch.tensor([1.0, s, s ** 2, s ** 3], dtype=torch.float64)
                weights = torch.tensor([s * (s - 1) * (s - 2) / -6, (s + 1) * (s - 1) * (s - 2) / 2,
                                        (s + 1) * s * (s - 2) / -2, (s + 1) * s * (s - 1) / 6], dtype=torch.float64)
                hv = (weights * h_tables[t, kk:kk + 4]).sum()
                kv = (weights * k_tables[t, kk:kk + 4]).sum()
                grad_xyz_ref[i] += hv / d * w * diff
                grad_xyz_ref[j] -= hv / d * w * diff
                grad_occ_ref[i] += kv * occupancy64[j]
                grad_occ_ref[j] += kv * occupancy64[i]

    for dtype, tolerance in ((torch.float64, 1e-10), (torch.float32, 1e-4)):
        xyz, occupancy = xyz64.to(device, dtype), occupancy64.to(device, dtype)
        hist = spread_pairs(jobs, xyz, occupancy, atom_indices, 3, n_nodes, dr, 0.0)
        grad_xyz, grad_occ = pair_gradients(xyz, occupancy, atom_indices, element_pairs, h_tables.to(device, dtype),
                                            k_tables.to(device, dtype), dr, 0.0, True, True)
        for name, got, want in (('moments', hist, hist_ref), ('position gradients', grad_xyz.view(n, 3), grad_xyz_ref),
                                ('occupancy gradients', grad_occ, grad_occ_ref)):
            error = ((got.cpu().double() - want).norm() / want.norm()).item()
            if not error < tolerance:
                raise RuntimeError(f'self-test failed for {name} in {dtype}: relative error {error:.1e}')


def library():
    """This module if the Triton kernels work on this machine (checked once), else None."""
    if not _state['checked']:
        _state['checked'] = True
        try:
            if triton is None:
                raise RuntimeError('Triton is not installed')
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA is not available')
            _self_test()
        except Exception as error:  # noqa: BLE001 - any failure means the PyTorch code is used
            _state['error'] = error
    return None if _state['error'] is not None else sys.modules[__name__]


def unavailable_reason():
    library()
    return _state['error']
