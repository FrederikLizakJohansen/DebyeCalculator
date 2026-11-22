//! Native CPU kernels for the grid pair sum of DebyeCalculator.
//!
//! Each function processes one batch ("job") of rows of an element pair, exactly as the per-batch loops of
//! `DebyeCalculator._spread_pairs` and `DebyeCalculator._grid_pair_sums_backward`, but in a single pass over the pairs
//! without batch-sized temporaries. Per-pair arithmetic is done in the calculation dtype (f32 or f64) in the same
//! operation order as the PyTorch code; accumulation is in f64. The GIL is released while a batch runs, so the
//! Python thread pool of DebyeCalculator runs batches in parallel.

// `!(d >= rthres)` is deliberate: like the PyTorch mask `d >= rthres`, it also drops NaN distances
#![allow(clippy::neg_cmp_op_on_partial_ord)]

use numpy::{Element, PyReadonlyArray1, PyReadwriteArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::ops::{Add, Div, Mul, Sub};

/// Version of the interface between DebyeCalculator and these kernels. DebyeCalculator only uses kernels whose
/// interface version it knows.
const INTERFACE_VERSION: u32 = 1;

trait Real:
    Copy
    + Send
    + Sync
    + PartialOrd
    + Element
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
{
    const ZERO: Self;
    const ONE: Self;
    fn sqrt(self) -> Self;
    fn floor(self) -> Self;
    fn from_f64(x: f64) -> Self;
    fn to_f64(self) -> f64;
    /// Truncating conversion of a value already checked to lie in [0, 2^32)
    fn to_index(self) -> u32;
}

macro_rules! impl_real {
    ($t:ty) => {
        impl Real for $t {
            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;
            #[inline(always)]
            fn sqrt(self) -> Self {
                <$t>::sqrt(self)
            }
            #[inline(always)]
            fn floor(self) -> Self {
                <$t>::floor(self)
            }
            #[inline(always)]
            fn from_f64(x: f64) -> Self {
                x as $t
            }
            #[inline(always)]
            fn to_f64(self) -> f64 {
                self as f64
            }
            #[inline(always)]
            fn to_index(self) -> u32 {
                self as u32
            }
        }
    };
}
impl_real!(f32);
impl_real!(f64);

/// Atom indices of a batch, checked against the number of atoms.
fn checked_indices(
    idx: &[i64],
    range: std::ops::Range<usize>,
    n_atoms: usize,
) -> Result<Vec<usize>, String> {
    idx.get(range)
        .ok_or("batch rows or columns out of range")?
        .iter()
        .map(|&i| {
            usize::try_from(i)
                .ok()
                .filter(|&i| i < n_atoms)
                .ok_or_else(|| format!("atom index {i} out of range"))
        })
        .collect()
}

/// One batch: rows of element a, and the columns of element b gathered into contiguous arrays.
struct Batch<T> {
    rows: Vec<usize>,
    cols: Vec<usize>,
    col_xyz: Vec<[T; 3]>,
    col_occupancy: Vec<T>,
    /// Element pair a == b: batch row r pairs with columns r.. only (the columns start at row start + 1), so j > i.
    same: bool,
}

impl<T: Real> Batch<T> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        xyz: &[T],
        occupancy: Option<&[T]>,
        idx_a: &[i64],
        idx_b: &[i64],
        start: usize,
        stop: usize,
        same: bool,
    ) -> Result<Self, String> {
        if !xyz.len().is_multiple_of(3) {
            return Err("xyz must hold 3 coordinates per atom".into());
        }
        let n_atoms = xyz.len() / 3;
        if start > stop {
            return Err("start must not exceed stop".into());
        }
        let col_start = if same { start + 1 } else { 0 };
        let rows = checked_indices(idx_a, start..stop, n_atoms)?;
        let cols = checked_indices(idx_b, col_start.min(idx_b.len())..idx_b.len(), n_atoms)?;
        let col_xyz = cols
            .iter()
            .map(|&j| [xyz[3 * j], xyz[3 * j + 1], xyz[3 * j + 2]])
            .collect();
        let col_occupancy =
            occupancy.map_or_else(Vec::new, |o| cols.iter().map(|&j| o[j]).collect());
        Ok(Batch {
            rows,
            cols,
            col_xyz,
            col_occupancy,
            same,
        })
    }

    /// First column (index into the column arrays) paired with batch row r.
    #[inline(always)]
    fn col_offset(&self, r: usize) -> usize {
        if self.same {
            r.min(self.cols.len())
        } else {
            0
        }
    }
}

/// Difference vector and distance. dr is a power of two, so the bin position u = d / dr, floor(u) and the fraction
/// s = u - floor(u) computed from it are exact, as in the PyTorch code. Distance: ((0 + dx^2) + dy^2) + dz^2 as torch.cdist(compute_mode='donot_use_mm_for_euclid_dist').
#[inline(always)]
fn diff<T: Real>(a: &[T; 3], b: &[T; 3]) -> (T, T, T, T) {
    let (dx, dy, dz) = (a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    (dx, dy, dz, (dx * dx + dy * dy + dz * dz).sqrt())
}

#[inline(always)]
fn atom<T: Real>(xyz: &[T], i: usize) -> [T; 3] {
    [xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]]
}

/// Columns per block: distances, bins and fractional positions of a block are computed in one vectorisable pass,
/// then scattered into the histogram.
const BLOCK: usize = 64;

fn spread<T: Real, const WEIGHTED: bool, const THRESHOLD: bool>(
    xyz: &[T],
    occupancy: Option<&[T]>,
    batch: &Batch<T>,
    inv_dr: T,
    rthres: T,
    hist: &mut [[f64; 4]],
) -> Result<(), String> {
    let n_bins = T::from_f64(hist.len().min(u32::MAX as usize) as f64);
    let mut bins = [0u32; BLOCK];
    let mut fractions = [T::ZERO; BLOCK];
    let mut weights = [T::ONE; BLOCK];
    for (r, &i) in batch.rows.iter().enumerate() {
        let xi = atom(xyz, i);
        let oi = occupancy.map_or(T::ONE, |o| o[i]);
        let first = batch.col_offset(r);
        let mut block_start = first;
        for block in batch.col_xyz[first..].chunks(BLOCK) {
            let n = block.len();
            let mut in_grid = true;
            for (t, xj) in block.iter().enumerate() {
                let (_, _, _, d) = diff(&xi, xj);
                let u = d * inv_dr;
                let b = u.floor();
                // Also false for NaN
                in_grid &= b >= T::ZERO && b < n_bins;
                bins[t] = b.to_index();
                fractions[t] = u - b;
                // Pairs below rthres get weight zero, as the mask (d >= rthres) of the PyTorch code
                let w = if WEIGHTED {
                    oi * batch.col_occupancy[block_start + t]
                } else {
                    T::ONE
                };
                weights[t] = if THRESHOLD && !(d >= rthres) {
                    T::ZERO
                } else {
                    w
                };
            }
            if !in_grid {
                return Err("pair distance outside the distance grid".into());
            }
            for t in 0..n {
                let (s, w) = (fractions[t], weights[t]);
                let s2 = s * s;
                let s3 = s2 * s;
                let h = &mut hist[bins[t] as usize];
                if WEIGHTED || THRESHOLD {
                    h[0] += w.to_f64();
                    h[1] += (s * w).to_f64();
                    h[2] += (s2 * w).to_f64();
                    h[3] += (s3 * w).to_f64();
                } else {
                    h[0] += 1.0;
                    h[1] += s.to_f64();
                    h[2] += s2.to_f64();
                    h[3] += s3.to_f64();
                }
            }
            block_start += n;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn backward<T: Real, const XYZ: bool, const OCC: bool, const THRESHOLD: bool>(
    xyz: &[T],
    occupancy: &[T],
    batch: &Batch<T>,
    inv_dr: T,
    rthres: T,
    h_table: &[T],
    k_table: &[T],
    grad_xyz: &mut [f64],
    grad_occupancy: &mut [f64],
) -> Result<(), String> {
    let n_bins = T::from_f64((h_table.len().max(k_table.len()) - 3) as f64);
    let (six, two) = (T::from_f64(6.0), T::from_f64(2.0));
    let (neg_six, neg_two) = (T::ZERO - six, T::ZERO - two);
    let interpolate = |t: &[T], k: usize, w: &[T; 4]| {
        let t = &t[k..k + 4];
        w[0] * t[0] + w[1] * t[1] + w[2] * t[2] + w[3] * t[3]
    };
    // Column gradients of the batch, scattered to the atoms at the end
    let n_cols = batch.cols.len();
    let mut col_grad_xyz = vec![[0.0f64; 3]; if XYZ { n_cols } else { 0 }];
    let mut col_grad_occupancy = vec![0.0f64; if OCC { n_cols } else { 0 }];

    let mut bins = [0u32; BLOCK];
    let mut distances = [T::ZERO; BLOCK];
    let mut stencils = [[T::ZERO; 4]; BLOCK];
    let mut included = [true; BLOCK];
    for (r, &i) in batch.rows.iter().enumerate() {
        let xi = atom(xyz, i);
        let oi = occupancy[i];
        let (mut gx, mut gy, mut gz, mut go) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        let first = batch.col_offset(r);
        let mut block_start = first;
        for block in batch.col_xyz[first..].chunks(BLOCK) {
            let n = block.len();
            // Distances, bins and cubic Lagrange weights of nodes k..k+3 (in the operation order of the PyTorch code)
            let mut in_grid = true;
            for (t, xj) in block.iter().enumerate() {
                let (_, _, _, d) = diff(&xi, xj);
                let u = d * inv_dr;
                let b = u.floor();
                // Also false for NaN
                in_grid &= b >= T::ZERO && b < n_bins;
                let s = u - b;
                let (sp1, sm1, sm2) = (s + T::ONE, s - T::ONE, s - two);
                bins[t] = b.to_index();
                distances[t] = d;
                stencils[t] = [
                    s * sm1 * sm2 / neg_six,
                    sp1 * sm1 * sm2 / two,
                    sp1 * s * sm2 / neg_two,
                    sp1 * s * sm1 / six,
                ];
                if THRESHOLD {
                    included[t] = d >= rthres;
                }
            }
            if !in_grid {
                return Err("pair distance outside the distance grid".into());
            }
            for t in 0..n {
                if THRESHOLD && !included[t] {
                    continue;
                }
                let (c, k, d, w) = (
                    block_start + t,
                    bins[t] as usize,
                    distances[t],
                    &stencils[t],
                );
                let oj = batch.col_occupancy[c];
                if XYZ && d > T::ZERO {
                    let (dx, dy, dz, _) = diff(&xi, &block[t]);
                    let coefficient = interpolate(h_table, k, w) / d * (oi * oj);
                    let (cx, cy, cz) = (
                        (coefficient * dx).to_f64(),
                        (coefficient * dy).to_f64(),
                        (coefficient * dz).to_f64(),
                    );
                    gx += cx;
                    gy += cy;
                    gz += cz;
                    let g = &mut col_grad_xyz[c];
                    g[0] -= cx;
                    g[1] -= cy;
                    g[2] -= cz;
                }
                if OCC {
                    let kv = interpolate(k_table, k, w);
                    go += (kv * oj).to_f64();
                    col_grad_occupancy[c] += (kv * oi).to_f64();
                }
            }
            block_start += n;
        }
        if XYZ {
            grad_xyz[3 * i] += gx;
            grad_xyz[3 * i + 1] += gy;
            grad_xyz[3 * i + 2] += gz;
        }
        if OCC {
            grad_occupancy[i] += go;
        }
    }
    for (c, &j) in batch.cols.iter().enumerate() {
        if XYZ {
            for (x, g) in col_grad_xyz[c].iter().enumerate() {
                grad_xyz[3 * j + x] += g;
            }
        }
        if OCC {
            grad_occupancy[j] += col_grad_occupancy[c];
        }
    }
    Ok(())
}

/// Run the spread kernel specialised for weights and threshold.
fn run_spread<T: Real>(
    xyz: &[T],
    occupancy: Option<&[T]>,
    batch: &Batch<T>,
    inv_dr: T,
    rthres: T,
    hist: &mut [[f64; 4]],
) -> Result<(), String> {
    match (occupancy.is_some(), rthres > T::ZERO) {
        (false, false) => spread::<T, false, false>(xyz, occupancy, batch, inv_dr, rthres, hist),
        (false, true) => spread::<T, false, true>(xyz, occupancy, batch, inv_dr, rthres, hist),
        (true, false) => spread::<T, true, false>(xyz, occupancy, batch, inv_dr, rthres, hist),
        (true, true) => spread::<T, true, true>(xyz, occupancy, batch, inv_dr, rthres, hist),
    }
}

/// Run the backward kernel specialised for the requested gradients and threshold.
#[allow(clippy::too_many_arguments)]
fn run_backward<T: Real>(
    xyz: &[T],
    occupancy: &[T],
    batch: &Batch<T>,
    inv_dr: T,
    rthres: T,
    h_table: Option<&[T]>,
    k_table: Option<&[T]>,
    grad_xyz: Option<&mut [f64]>,
    grad_occupancy: Option<&mut [f64]>,
) -> Result<(), String> {
    let (h, k) = (h_table.unwrap_or(&[]), k_table.unwrap_or(&[]));
    let (mut empty_xyz, mut empty_occupancy) = ([0.0f64; 0], [0.0f64; 0]);
    let (gx, go) = (
        grad_xyz.unwrap_or(&mut empty_xyz),
        grad_occupancy.unwrap_or(&mut empty_occupancy),
    );
    macro_rules! run {
        ($xyz:literal, $occ:literal) => {
            if rthres > T::ZERO {
                backward::<T, $xyz, $occ, true>(xyz, occupancy, batch, inv_dr, rthres, h, k, gx, go)
            } else {
                backward::<T, $xyz, $occ, false>(
                    xyz, occupancy, batch, inv_dr, rthres, h, k, gx, go,
                )
            }
        };
    }
    match (h_table.is_some(), k_table.is_some()) {
        (true, true) => run!(true, true),
        (true, false) => run!(true, false),
        (false, true) => run!(false, true),
        (false, false) => Ok(()),
    }
}

fn check_len(name: &str, len: usize, expected: usize) -> PyResult<()> {
    if len != expected {
        return Err(PyValueError::new_err(format!(
            "{name} has length {len}, expected {expected}"
        )));
    }
    Ok(())
}

macro_rules! define_kernels {
    ($t:ty, $spread:ident, $backward:ident) => {
        /// Accumulate the bin moments sum(w), sum(w s), sum(w s^2), sum(w s^3) of one batch into hist
        /// (flattened (n_nodes, 4), float64). xyz is flattened (n_atoms, 3); occupancy is None for uniform occupancy.
        #[pyfunction]
        #[pyo3(signature = (xyz, occupancy, idx_a, idx_b, start, stop, same, inv_dr, rthres, hist))]
        #[allow(clippy::too_many_arguments)]
        fn $spread(
            py: Python<'_>,
            xyz: PyReadonlyArray1<'_, $t>,
            occupancy: Option<PyReadonlyArray1<'_, $t>>,
            idx_a: PyReadonlyArray1<'_, i64>,
            idx_b: PyReadonlyArray1<'_, i64>,
            start: usize,
            stop: usize,
            same: bool,
            inv_dr: f64,
            rthres: f64,
            mut hist: PyReadwriteArray1<'_, f64>,
        ) -> PyResult<()> {
            let xyz = xyz.as_slice()?;
            let occupancy = occupancy.as_ref().map(|o| o.as_slice()).transpose()?;
            if let Some(o) = occupancy {
                check_len("occupancy", o.len(), xyz.len() / 3)?;
            }
            let hist = hist.as_slice_mut()?;
            let (hist, rest) = hist.as_chunks_mut::<4>();
            if !rest.is_empty() {
                return Err(PyValueError::new_err("hist must have 4 moments per node"));
            }
            let (idx_a, idx_b) = (idx_a.as_slice()?, idx_b.as_slice()?);
            let (inv_dr, rthres) = (<$t>::from_f64(inv_dr), <$t>::from_f64(rthres));
            py.detach(|| {
                let batch = Batch::new(xyz, occupancy, idx_a, idx_b, start, stop, same)?;
                run_spread(xyz, occupancy, &batch, inv_dr, rthres, hist)
            })
            .map_err(PyValueError::new_err)
        }

        /// Gradients of one batch: accumulate into grad_xyz (flattened (n_atoms, 3)) and grad_occupancy (n_atoms),
        /// both float64, from the node tables h and k of the element pair (length n_nodes + 3). A gradient that is
        /// not needed is passed as None together with its table.
        #[pyfunction]
        #[pyo3(signature = (xyz, occupancy, idx_a, idx_b, start, stop, same, inv_dr, rthres, h_table, k_table, grad_xyz, grad_occupancy))]
        #[allow(clippy::too_many_arguments)]
        fn $backward(
            py: Python<'_>,
            xyz: PyReadonlyArray1<'_, $t>,
            occupancy: PyReadonlyArray1<'_, $t>,
            idx_a: PyReadonlyArray1<'_, i64>,
            idx_b: PyReadonlyArray1<'_, i64>,
            start: usize,
            stop: usize,
            same: bool,
            inv_dr: f64,
            rthres: f64,
            h_table: Option<PyReadonlyArray1<'_, $t>>,
            k_table: Option<PyReadonlyArray1<'_, $t>>,
            mut grad_xyz: Option<PyReadwriteArray1<'_, f64>>,
            mut grad_occupancy: Option<PyReadwriteArray1<'_, f64>>,
        ) -> PyResult<()> {
            let xyz = xyz.as_slice()?;
            let occupancy = occupancy.as_slice()?;
            check_len("occupancy", occupancy.len(), xyz.len() / 3)?;
            let h_table = h_table.as_ref().map(|t| t.as_slice()).transpose()?;
            let k_table = k_table.as_ref().map(|t| t.as_slice()).transpose()?;
            let grad_xyz = grad_xyz.as_mut().map(|g| g.as_slice_mut()).transpose()?;
            let grad_occupancy = grad_occupancy.as_mut().map(|g| g.as_slice_mut()).transpose()?;
            if grad_xyz.is_some() != h_table.is_some() || grad_occupancy.is_some() != k_table.is_some() {
                return Err(PyValueError::new_err("pass each gradient together with its node table"));
            }
            if let Some(g) = &grad_xyz {
                check_len("grad_xyz", g.len(), xyz.len())?;
            }
            if let Some(g) = &grad_occupancy {
                check_len("grad_occupancy", g.len(), occupancy.len())?;
            }
            let n_table = h_table.or(k_table).map_or(0, |t| t.len());
            if h_table.is_some_and(|t| t.len() != n_table) || k_table.is_some_and(|t| t.len() != n_table)
                || (n_table < 4 && (h_table.is_some() || k_table.is_some()))
            {
                return Err(PyValueError::new_err("node tables must have the same length of at least 4"));
            }
            let (idx_a, idx_b) = (idx_a.as_slice()?, idx_b.as_slice()?);
            let (inv_dr, rthres) = (<$t>::from_f64(inv_dr), <$t>::from_f64(rthres));
            py.detach(|| {
                let batch = Batch::new(xyz, Some(occupancy), idx_a, idx_b, start, stop, same)?;
                run_backward(xyz, occupancy, &batch, inv_dr, rthres, h_table, k_table, grad_xyz, grad_occupancy)
            })
            .map_err(PyValueError::new_err)
        }
    };
}

define_kernels!(f32, spread_pairs_f32, grid_backward_f32);
define_kernels!(f64, spread_pairs_f64, grid_backward_f64);

#[pymodule]
fn debyecalculator_kernels(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("INTERFACE_VERSION", INTERFACE_VERSION)?;
    m.add_function(wrap_pyfunction!(spread_pairs_f32, m)?)?;
    m.add_function(wrap_pyfunction!(spread_pairs_f64, m)?)?;
    m.add_function(wrap_pyfunction!(grid_backward_f32, m)?)?;
    m.add_function(wrap_pyfunction!(grid_backward_f64, m)?)?;
    Ok(())
}
