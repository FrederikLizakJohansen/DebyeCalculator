"""
Native CPU kernels for the grid pair sum of DebyeCalculator, in C (prototype alongside the Rust kernels in
debyecalculator_kernels). Same functions and arguments as debyecalculator_kernels; the shared library is called through
ctypes, which releases the GIL during each call.
"""
import ctypes
import glob
import os

import numpy as np

__version__ = '1.1.0'

_here = os.path.dirname(os.path.abspath(__file__))
_candidates = sorted(glob.glob(os.path.join(_here, '_ckernels*')))
_candidates = [p for p in _candidates if p.endswith(('.so', '.dylib', '.pyd', '.dll'))]
if not _candidates:
    raise ImportError('debyecalculator_ckernels: the compiled library is missing; install with pip install ./kernels_c')
_lib = ctypes.CDLL(_candidates[0])

INTERFACE_VERSION = _lib.dc_interface_version()

_P, _I64, _D, _INT = ctypes.c_void_p, ctypes.c_int64, ctypes.c_double, ctypes.c_int
_ERRORS = {
    1: 'atom index out of range',
    2: 'pair distance outside the distance grid',
    3: 'out of memory',
    4: 'invalid batch range',
}
for _suffix in ('f32', 'f64'):
    _spread = getattr(_lib, 'dc_spread_' + _suffix)
    _spread.argtypes = [_P, _I64, _P, _P, _I64, _P, _I64, _I64, _I64, _INT, _D, _D, _P, _I64]
    _spread.restype = _INT
    _backward = getattr(_lib, 'dc_backward_' + _suffix)
    _backward.argtypes = [_P, _I64, _P, _P, _I64, _P, _I64, _I64, _I64, _INT, _D, _D, _P, _P, _I64, _P, _P]
    _backward.restype = _INT


def _array(a, dtype, name, writable=False):
    if a is None:
        return None
    if not isinstance(a, np.ndarray) or a.dtype != dtype or a.ndim != 1 or not a.flags.c_contiguous:
        raise TypeError(f'{name} must be a contiguous 1-D numpy array of {np.dtype(dtype)}')
    if writable and not a.flags.writeable:
        raise ValueError(f'{name} must be writable')
    return a


def _ptr(a):
    return None if a is None else a.ctypes.data


def _check(status):
    if status != 0:
        raise ValueError(_ERRORS.get(status, f'error {status}'))


def _make(dtype, suffix):
    spread_c = getattr(_lib, 'dc_spread_' + suffix)
    backward_c = getattr(_lib, 'dc_backward_' + suffix)

    def spread_pairs(xyz, occupancy, idx_a, idx_b, start, stop, same, inv_dr, rthres, hist):
        xyz = _array(xyz, dtype, 'xyz')
        occupancy = _array(occupancy, dtype, 'occupancy')
        idx_a, idx_b = _array(idx_a, np.int64, 'idx_a'), _array(idx_b, np.int64, 'idx_b')
        hist = _array(hist, np.float64, 'hist', writable=True)
        if xyz.size % 3 or hist.size % 4:
            raise ValueError('xyz must hold 3 coordinates per atom and hist 4 moments per node')
        if occupancy is not None and occupancy.size != xyz.size // 3:
            raise ValueError('occupancy must hold one value per atom')
        _check(spread_c(_ptr(xyz), xyz.size // 3, _ptr(occupancy), _ptr(idx_a), idx_a.size, _ptr(idx_b), idx_b.size,
                        int(start), int(stop), int(bool(same)), float(inv_dr), float(rthres), _ptr(hist), hist.size // 4))

    def grid_backward(xyz, occupancy, idx_a, idx_b, start, stop, same, inv_dr, rthres, h_table, k_table, grad_xyz,
                      grad_occupancy):
        xyz, occupancy = _array(xyz, dtype, 'xyz'), _array(occupancy, dtype, 'occupancy')
        idx_a, idx_b = _array(idx_a, np.int64, 'idx_a'), _array(idx_b, np.int64, 'idx_b')
        h_table, k_table = _array(h_table, dtype, 'h_table'), _array(k_table, dtype, 'k_table')
        grad_xyz = _array(grad_xyz, np.float64, 'grad_xyz', writable=True)
        grad_occupancy = _array(grad_occupancy, np.float64, 'grad_occupancy', writable=True)
        if (grad_xyz is None) != (h_table is None) or (grad_occupancy is None) != (k_table is None):
            raise ValueError('pass each gradient together with its node table')
        if xyz.size % 3 or occupancy.size != xyz.size // 3:
            raise ValueError('xyz must hold 3 coordinates and occupancy one value per atom')
        if (grad_xyz is not None and grad_xyz.size != xyz.size) or (
                grad_occupancy is not None and grad_occupancy.size != occupancy.size):
            raise ValueError('gradients must have the shapes of xyz and occupancy')
        tables = [t for t in (h_table, k_table) if t is not None]
        if not tables:
            return
        if len({t.size for t in tables}) != 1:
            raise ValueError('node tables must have the same length')
        _check(backward_c(_ptr(xyz), xyz.size // 3, _ptr(occupancy), _ptr(idx_a), idx_a.size, _ptr(idx_b), idx_b.size,
                          int(start), int(stop), int(bool(same)), float(inv_dr), float(rthres), _ptr(h_table),
                          _ptr(k_table), tables[0].size, _ptr(grad_xyz), _ptr(grad_occupancy)))

    return spread_pairs, grid_backward


spread_pairs_f32, grid_backward_f32 = _make(np.float32, 'f32')
spread_pairs_f64, grid_backward_f64 = _make(np.float64, 'f64')
