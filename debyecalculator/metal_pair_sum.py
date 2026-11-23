"""
Fused Metal kernels for the grid pair sum on Apple GPUs (device='mps', float32), compiled at run time with
torch.mps.compile_shader (PyTorch 2.6 or later); no build step.

The kernels do the per-pair work of DebyeCalculator._spread_pairs (forward) and
DebyeCalculator._grid_pair_sums_backward (backward) in one pass over the pairs, without batch-sized temporaries:

- forward: each thread takes one row atom i and a segment of the column atoms j, and adds the bin moments
  w, w s, w s^2, w s^3 of each pair to a histogram in 64-bit fixed point (two 32-bit atomics with carry). Each moment
  is rounded once to a multiple of 1/scale (scale = 2^31 / max w); the sums themselves are exact for any number of
  atoms, unlike float32 atomic adds. Moments are non-negative, so negative occupancies use the PyTorch code.
- backward: each thread takes one atom i and a segment of the atoms j of the other element and sums its own gradient
  terms over ordered pairs (i, j), with compensated summation; one atomic add per thread and segment.

The per-pair arithmetic is in float32 in the operation order of the PyTorch code, without fast math or contraction.
"""
import math

import torch

SOURCE = r"""
#include <metal_stdlib>
using namespace metal;
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)

// Add v >= 0, rounded to a multiple of 1/scale, to the 64-bit fixed-point number (h[0] low, h[1] high word)
inline void add_fixed(device atomic<uint>* h, float v, float scale)
{
    const uint q = (uint)rint(v * scale);
    const uint old = atomic_fetch_add_explicit(h, q, memory_order_relaxed);
    if (old > 0xFFFFFFFFu - q) atomic_fetch_add_explicit(h + 1, 1u, memory_order_relaxed);
}

// Bin moments of the pairs (row atom, column segment); threads = [n_rows, n_segments]
kernel void spread_pairs(
    device const float* xyz,
    device const float* occupancy,
    device const long* idx_a,
    device const long* idx_b,
    device atomic<uint>* hist,
    device atomic<int>* error,
    constant long& row_start,
    constant long& n_rows,
    constant long& n_b,
    constant long& same,
    constant long& weighted,
    constant long& n_segments,
    constant float& inv_dr,
    constant float& rthres,
    constant long& n_bins,
    constant float& scale,
    uint2 pos [[thread_position_in_grid]])
{
    const long r = pos.x;
    if (r >= n_rows) return;
    const long row = row_start + r;
    const long i = idx_a[row];
    const float xi = xyz[3 * i], yi = xyz[3 * i + 1], zi = xyz[3 * i + 2];
    const float oi = weighted ? occupancy[i] : 1.0f;
    const long length = (n_b + n_segments - 1) / n_segments;
    long c0 = (long)pos.y * length;
    const long c1 = min(c0 + length, n_b);
    if (same) c0 = max(c0, row + 1);
    const float limit = (float)n_bins;
    for (long c = c0; c < c1; c++) {
        const long j = idx_b[c];
        const float dx = xi - xyz[3 * j], dy = yi - xyz[3 * j + 1], dz = zi - xyz[3 * j + 2];
        const float d = sqrt(dx * dx + dy * dy + dz * dz);
        if (rthres > 0.0f && !(d >= rthres)) continue;
        const float u = d * inv_dr;
        const float b = floor(u);
        if (!(b >= 0.0f && b < limit)) {
            atomic_store_explicit(error, 1, memory_order_relaxed);
            continue;
        }
        const long k = (long)b;
        const float s = u - b;
        const float s2 = s * s;
        const float s3 = s2 * s;
        const float w = weighted ? oi * occupancy[j] : 1.0f;
        device atomic<uint>* h = hist + 8 * k;
        add_fixed(h, w, scale);
        add_fixed(h + 2, s * w, scale);
        add_fixed(h + 4, s2 * w, scale);
        add_fixed(h + 6, s3 * w, scale);
    }
}

// Compensated (Kahan) summation in float32
struct Kahan {
    float sum = 0.0f, c = 0.0f;
    void add(float x) { const float y = x - c; const float t = sum + y; c = (t - sum) - y; sum = t; }
};

// Gradient terms of atom i over the ordered pairs (i, j) of a column segment; threads = [n_rows, n_segments].
// With skip_self (one element), the pair j == i is left out.
kernel void pair_gradients(
    device const float* xyz,
    device const float* occupancy,
    device const long* idx_a,
    device const long* idx_b,
    device const float* h_table,
    device const float* k_table,
    device atomic<float>* grad_xyz,
    device atomic<float>* grad_occupancy,
    device atomic<int>* error,
    constant long& row_start,
    constant long& n_rows,
    constant long& n_b,
    constant long& skip_self,
    constant long& need_xyz,
    constant long& need_occupancy,
    constant long& n_segments,
    constant float& inv_dr,
    constant float& rthres,
    constant long& n_bins,
    uint2 pos [[thread_position_in_grid]])
{
    const long r = pos.x;
    if (r >= n_rows) return;
    const long i = idx_a[row_start + r];
    const float xi = xyz[3 * i], yi = xyz[3 * i + 1], zi = xyz[3 * i + 2];
    const float oi = occupancy[i];
    const long length = (n_b + n_segments - 1) / n_segments;
    const long c0 = (long)pos.y * length;
    const long c1 = min(c0 + length, n_b);
    const float limit = (float)n_bins;
    Kahan gx, gy, gz, go;
    for (long c = c0; c < c1; c++) {
        const long j = idx_b[c];
        if (skip_self && j == i) continue;
        const float dx = xi - xyz[3 * j], dy = yi - xyz[3 * j + 1], dz = zi - xyz[3 * j + 2];
        const float d = sqrt(dx * dx + dy * dy + dz * dz);
        if (rthres > 0.0f && !(d >= rthres)) continue;
        const float u = d * inv_dr;
        const float b = floor(u);
        if (!(b >= 0.0f && b < limit)) {
            atomic_store_explicit(error, 1, memory_order_relaxed);
            continue;
        }
        const long k = (long)b;
        const float s = u - b, sp1 = s + 1.0f, sm1 = s - 1.0f, sm2 = s - 2.0f;
        const float w0 = s * sm1 * sm2 / -6.0f, w1 = sp1 * sm1 * sm2 / 2.0f;
        const float w2 = sp1 * s * sm2 / -2.0f, w3 = sp1 * s * sm1 / 6.0f;
        const float oj = occupancy[j];
        if (need_xyz && d > 0.0f) {
            const float hv = w0 * h_table[k] + w1 * h_table[k + 1] + w2 * h_table[k + 2] + w3 * h_table[k + 3];
            const float coefficient = hv / d * (oi * oj);
            gx.add(coefficient * dx);
            gy.add(coefficient * dy);
            gz.add(coefficient * dz);
        }
        if (need_occupancy) {
            const float kv = w0 * k_table[k] + w1 * k_table[k + 1] + w2 * k_table[k + 2] + w3 * k_table[k + 3];
            go.add(kv * oj);
        }
    }
    if (need_xyz) {
        atomic_fetch_add_explicit(grad_xyz + 3 * i, gx.sum, memory_order_relaxed);
        atomic_fetch_add_explicit(grad_xyz + 3 * i + 1, gy.sum, memory_order_relaxed);
        atomic_fetch_add_explicit(grad_xyz + 3 * i + 2, gz.sum, memory_order_relaxed);
    }
    if (need_occupancy) {
        atomic_fetch_add_explicit(grad_occupancy + i, go.sum, memory_order_relaxed);
    }
}
"""

# Threads per dispatch to aim for, and pairs per backward dispatch
TARGET_THREADS = 1 << 16
MIN_SEGMENT = 32
PAIRS_PER_DISPATCH = 1 << 27

_library = None
_unavailable = None


def library():
    """The compiled kernels, or None if Metal kernels can't be compiled here."""
    global _library, _unavailable
    if _library is None and _unavailable is None:
        try:
            if not (hasattr(torch, 'mps') and hasattr(torch.mps, 'compile_shader') and torch.backends.mps.is_available()):
                raise RuntimeError('torch.mps.compile_shader is not available')
            _library = torch.mps.compile_shader(SOURCE)
        except Exception as error:  # noqa: BLE001 - any failure means the PyTorch code is used
            _unavailable = error
    return _library


def _segments(n_rows, n_cols):
    """Column segments per row, so that a dispatch has about TARGET_THREADS threads."""
    return max(1, min(math.ceil(TARGET_THREADS / max(1, n_rows)), math.ceil(n_cols / MIN_SEGMENT)))


def _check(error):
    if error.item():
        raise ValueError('pair distance outside the distance grid')


def applies(occupancy):
    """Whether the kernels can be used: the fixed-point histogram needs non-negative pair weights."""
    return library() is not None and not bool((occupancy < 0).any())


def spread_pairs(jobs, xyz, occupancy, atom_indices, n_element_pairs, n_nodes, dr, rthres, progress=None):
    """
    Bin moments of the given batches, as DebyeCalculator._spread_pairs. Returns a float32 tensor on the device of
    shape (n_element_pairs, n_nodes, 4).
    """
    lib = library()
    device = xyz.device
    xyz = xyz.detach().contiguous().reshape(-1)
    weighted = not bool(torch.all(occupancy == 1))
    occupancy = occupancy.detach().contiguous()
    indices = [idx.to(device=device, dtype=torch.int64).contiguous() for idx in atom_indices]
    # Fixed point: largest power of two with max(w) * scale <= 2^31 (moments are at most w)
    max_weight = float(occupancy.max()) ** 2 if weighted and occupancy.numel() else 1.0
    scale = 2.0 ** min(31, math.floor(31 - math.log2(max(max_weight, 2.0 ** -60))))
    hist = torch.zeros((n_element_pairs, n_nodes * 8), device=device, dtype=torch.int32)
    error = torch.zeros(1, device=device, dtype=torch.int32)

    for pair_number, a, b, start, stop in jobs:
        n_rows, n_b = stop - start, indices[b].numel()
        n_segments = _segments(n_rows, n_b)
        lib.spread_pairs(xyz, occupancy, indices[a], indices[b], hist[pair_number], error, start, n_rows, n_b,
                         int(a == b), int(weighted), n_segments, 1.0 / dr, float(rthres), n_nodes, scale,
                         threads=[n_rows, n_segments])
        if progress is not None:
            progress.step()

    _check(error)
    words = hist.cpu().to(torch.int64) & 0xFFFFFFFF
    total = (words[:, 1::2] * 2 ** 32 + words[:, 0::2]).double() / scale
    return total.to(device=device, dtype=torch.float32).view(n_element_pairs, n_nodes, 4)


def pair_gradients(xyz, occupancy, atom_indices, element_pairs, h_tables, k_tables, dr, rthres, need_xyz,
                   need_occupancy):
    """
    Gradients with respect to positions and occupancies, as DebyeCalculator._grid_pair_sums_backward, from the node
    tables (n_element_pairs, n_nodes + 3) in float32. Returns float32 tensors on the device (None if not needed).
    """
    lib = library()
    device = xyz.device
    xyz = xyz.detach().to(torch.float32).contiguous().reshape(-1)
    occupancy = occupancy.detach().to(torch.float32).contiguous()
    indices = [idx.to(device=device, dtype=torch.int64).contiguous() for idx in atom_indices]
    grad_xyz = torch.zeros(xyz.numel(), device=device, dtype=torch.float32)
    grad_occupancy = torch.zeros(occupancy.numel(), device=device, dtype=torch.float32)
    error = torch.zeros(1, device=device, dtype=torch.int32)
    dummy = torch.zeros(4, device=device, dtype=torch.float32)

    for pair_number, (a, b) in enumerate(element_pairs):
        h = h_tables[pair_number].contiguous() if need_xyz else dummy
        k = k_tables[pair_number].contiguous() if need_occupancy else dummy
        n_bins = (h if need_xyz else k).numel() - 3
        # Ordered pairs: rows of a against columns of b and, for two elements, rows of b against columns of a
        for rows, cols in ([(a, b)] if a == b else [(a, b), (b, a)]):
            n_a, n_b = indices[rows].numel(), indices[cols].numel()
            rows_per_dispatch = max(1, PAIRS_PER_DISPATCH // max(1, n_b))
            for start in range(0, n_a, rows_per_dispatch):
                n_rows = min(rows_per_dispatch, n_a - start)
                n_segments = _segments(n_rows, n_b)
                lib.pair_gradients(xyz, occupancy, indices[rows], indices[cols], h, k, grad_xyz, grad_occupancy, error,
                                   start, n_rows, n_b, int(a == b), int(need_xyz), int(need_occupancy), n_segments,
                                   1.0 / dr, float(rthres), n_bins, threads=[n_rows, n_segments])
    _check(error)
    return (grad_xyz if need_xyz else None), (grad_occupancy if need_occupancy else None)
