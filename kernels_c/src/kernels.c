/*
 * Native CPU kernels for the grid pair sum of DebyeCalculator, in C (prototype alongside the Rust kernels in
 * ../kernels; same algorithm and interface). Each call processes one batch of rows of an element pair, exactly as
 * the per-batch loops of DebyeCalculator._spread_pairs and DebyeCalculator._grid_pair_sums_backward, in a single pass
 * over the pairs. Per-pair arithmetic is in the calculation type (float or double) in the operation order of the
 * PyTorch code; accumulation is in double. Compile without floating-point contraction (-ffp-contract=off).
 *
 * Return codes: 0 ok, 1 atom index out of range, 2 pair distance outside the distance grid, 3 out of memory,
 * 4 invalid batch range.
 */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#if defined(_WIN32)
#define DC_EXPORT __declspec(dllexport)
#else
#define DC_EXPORT __attribute__((visibility("default")))
#endif

#define DC_OK 0
#define DC_BAD_INDEX 1
#define DC_OUT_OF_GRID 2
#define DC_NO_MEMORY 3
#define DC_BAD_RANGE 4

#define BLOCK 64

DC_EXPORT int dc_interface_version(void) { return 1; }

#define DEFINE_KERNELS(T, SUFFIX, SQRT, FLOOR)                                                                        \
                                                                                                                      \
    /* Rows of element a and the gathered columns of element b of one batch */                                       \
    typedef struct {                                                                                                  \
        int64_t n_rows, n_cols;                                                                                       \
        const int64_t *rows, *cols;                                                                                   \
        T *col_xyz, *col_occ;                                                                                         \
        int same;                                                                                                     \
    } batch_##SUFFIX;                                                                                                 \
                                                                                                                      \
    static int batch_init_##SUFFIX(batch_##SUFFIX *b, const T *xyz, int64_t n_atoms, const T *occ,                    \
                                   const int64_t *idx_a, int64_t n_a, const int64_t *idx_b, int64_t n_b,              \
                                   int64_t start, int64_t stop, int same) {                                           \
        int64_t col_start = same ? start + 1 : 0;                                                                     \
        if (start < 0 || start > stop || stop > n_a) return DC_BAD_RANGE;                                             \
        if (col_start > n_b) col_start = n_b;                                                                         \
        b->n_rows = stop - start;                                                                                     \
        b->n_cols = n_b - col_start;                                                                                  \
        b->rows = idx_a + start;                                                                                      \
        b->cols = idx_b + col_start;                                                                                  \
        b->same = same;                                                                                               \
        for (int64_t r = 0; r < b->n_rows; r++)                                                                       \
            if (b->rows[r] < 0 || b->rows[r] >= n_atoms) return DC_BAD_INDEX;                                         \
        b->col_xyz = (T *)malloc(sizeof(T) * (size_t)(3 * b->n_cols + 1));                                            \
        b->col_occ = (T *)malloc(sizeof(T) * (size_t)(b->n_cols + 1));                                                \
        if (!b->col_xyz || !b->col_occ) return DC_NO_MEMORY;                                                          \
        for (int64_t c = 0; c < b->n_cols; c++) {                                                                     \
            int64_t j = b->cols[c];                                                                                   \
            if (j < 0 || j >= n_atoms) return DC_BAD_INDEX;                                                           \
            b->col_xyz[3 * c] = xyz[3 * j];                                                                           \
            b->col_xyz[3 * c + 1] = xyz[3 * j + 1];                                                                   \
            b->col_xyz[3 * c + 2] = xyz[3 * j + 2];                                                                   \
            b->col_occ[c] = occ ? occ[j] : (T)1;                                                                      \
        }                                                                                                             \
        return DC_OK;                                                                                                 \
    }                                                                                                                 \
                                                                                                                      \
    static void batch_free_##SUFFIX(batch_##SUFFIX *b) {                                                              \
        free(b->col_xyz);                                                                                             \
        free(b->col_occ);                                                                                             \
    }                                                                                                                 \
                                                                                                                      \
    /* Moments sum(w), sum(w s), sum(w s^2), sum(w s^3) per bin, into hist[n_bins][4] */                              \
    DC_EXPORT int dc_spread_##SUFFIX(const T *xyz, int64_t n_atoms, const T *occ, const int64_t *idx_a, int64_t n_a,  \
                                     const int64_t *idx_b, int64_t n_b, int64_t start, int64_t stop, int same,        \
                                     double inv_dr_, double rthres_, double *hist, int64_t n_bins_) {                 \
        batch_##SUFFIX b = {0};                                                                                       \
        int status = batch_init_##SUFFIX(&b, xyz, n_atoms, occ, idx_a, n_a, idx_b, n_b, start, stop, same);           \
        const T inv_dr = (T)inv_dr_, rthres = (T)rthres_, n_bins = (T)n_bins_;                                        \
        const int weighted = occ != NULL, threshold = rthres > 0;                                                     \
        uint32_t bins[BLOCK];                                                                                         \
        T fractions[BLOCK], weights[BLOCK];                                                                           \
        for (int64_t r = 0; status == DC_OK && r < b.n_rows; r++) {                                                   \
            const int64_t i = b.rows[r];                                                                              \
            const T xi = xyz[3 * i], yi = xyz[3 * i + 1], zi = xyz[3 * i + 2];                                        \
            const T oi = weighted ? occ[i] : (T)1;                                                                    \
            const int64_t first = b.same ? (r < b.n_cols ? r : b.n_cols) : 0;                                         \
            for (int64_t c0 = first; c0 < b.n_cols; c0 += BLOCK) {                                                    \
                const int64_t n = b.n_cols - c0 < BLOCK ? b.n_cols - c0 : BLOCK;                                      \
                int in_grid = 1;                                                                                      \
                for (int64_t t = 0; t < n; t++) {                                                                     \
                    const T *xj = b.col_xyz + 3 * (c0 + t);                                                           \
                    const T dx = xi - xj[0], dy = yi - xj[1], dz = zi - xj[2];                                        \
                    const T d = SQRT(dx * dx + dy * dy + dz * dz);                                                    \
                    const T u = d * inv_dr;                                                                           \
                    const T bb = FLOOR(u);                                                                            \
                    in_grid &= (bb >= 0) & (bb < n_bins);                                                             \
                    bins[t] = in_grid ? (uint32_t)bb : 0;                                                             \
                    fractions[t] = u - bb;                                                                            \
                    const T w = weighted ? oi * b.col_occ[c0 + t] : (T)1;                                             \
                    weights[t] = (threshold && !(d >= rthres)) ? (T)0 : w;                                            \
                }                                                                                                     \
                if (!in_grid) {                                                                                       \
                    status = DC_OUT_OF_GRID;                                                                          \
                    break;                                                                                            \
                }                                                                                                     \
                for (int64_t t = 0; t < n; t++) {                                                                     \
                    const T s = fractions[t], w = weights[t];                                                         \
                    const T s2 = s * s;                                                                               \
                    const T s3 = s2 * s;                                                                              \
                    double *h = hist + 4 * (int64_t)bins[t];                                                          \
                    h[0] += (double)w;                                                                                \
                    h[1] += (double)(s * w);                                                                          \
                    h[2] += (double)(s2 * w);                                                                         \
                    h[3] += (double)(s3 * w);                                                                         \
                }                                                                                                     \
            }                                                                                                         \
        }                                                                                                             \
        batch_free_##SUFFIX(&b);                                                                                      \
        return status;                                                                                                \
    }                                                                                                                 \
                                                                                                                      \
    /* Gradients of one batch into grad_xyz[n_atoms*3] and grad_occ[n_atoms] (either NULL with its table) */          \
    DC_EXPORT int dc_backward_##SUFFIX(const T *xyz, int64_t n_atoms, const T *occ, const int64_t *idx_a,             \
                                       int64_t n_a, const int64_t *idx_b, int64_t n_b, int64_t start, int64_t stop,   \
                                       int same, double inv_dr_, double rthres_, const T *h_table, const T *k_table,  \
                                       int64_t n_table, double *grad_xyz, double *grad_occ) {                         \
        if (n_table < 4) return DC_BAD_RANGE;                                                                         \
        batch_##SUFFIX b = {0};                                                                                       \
        int status = batch_init_##SUFFIX(&b, xyz, n_atoms, occ, idx_a, n_a, idx_b, n_b, start, stop, same);           \
        const T inv_dr = (T)inv_dr_, rthres = (T)rthres_, n_bins = (T)(n_table - 3);                                  \
        const T six = (T)6, two = (T)2, neg_six = (T)-6, neg_two = (T)-2, one = (T)1;                                 \
        const int need_xyz = grad_xyz != NULL && h_table != NULL;                                                     \
        const int need_occ = grad_occ != NULL && k_table != NULL;                                                     \
        const int threshold = rthres > 0;                                                                             \
        double *col_gxyz = NULL, *col_gocc = NULL;                                                                    \
        if (status == DC_OK) {                                                                                        \
            col_gxyz = need_xyz ? (double *)calloc((size_t)(3 * b.n_cols + 1), sizeof(double)) : NULL;                \
            col_gocc = need_occ ? (double *)calloc((size_t)(b.n_cols + 1), sizeof(double)) : NULL;                    \
            if ((need_xyz && !col_gxyz) || (need_occ && !col_gocc)) status = DC_NO_MEMORY;                            \
        }                                                                                                             \
        uint32_t bins[BLOCK];                                                                                         \
        T distances[BLOCK], st[BLOCK][4];                                                                             \
        int included[BLOCK];                                                                                          \
        for (int64_t r = 0; status == DC_OK && r < b.n_rows; r++) {                                                   \
            const int64_t i = b.rows[r];                                                                              \
            const T xi = xyz[3 * i], yi = xyz[3 * i + 1], zi = xyz[3 * i + 2];                                        \
            const T oi = occ[i];                                                                                      \
            double gx = 0, gy = 0, gz = 0, go = 0;                                                                    \
            const int64_t first = b.same ? (r < b.n_cols ? r : b.n_cols) : 0;                                         \
            for (int64_t c0 = first; c0 < b.n_cols; c0 += BLOCK) {                                                    \
                const int64_t n = b.n_cols - c0 < BLOCK ? b.n_cols - c0 : BLOCK;                                      \
                int in_grid = 1;                                                                                      \
                for (int64_t t = 0; t < n; t++) {                                                                     \
                    const T *xj = b.col_xyz + 3 * (c0 + t);                                                           \
                    const T dx = xi - xj[0], dy = yi - xj[1], dz = zi - xj[2];                                        \
                    const T d = SQRT(dx * dx + dy * dy + dz * dz);                                                    \
                    const T u = d * inv_dr;                                                                           \
                    const T bb = FLOOR(u);                                                                            \
                    in_grid &= (bb >= 0) & (bb < n_bins);                                                             \
                    const T s = u - bb, sp1 = s + one, sm1 = s - one, sm2 = s - two;                                  \
                    bins[t] = in_grid ? (uint32_t)bb : 0;                                                             \
                    distances[t] = d;                                                                                 \
                    st[t][0] = s * sm1 * sm2 / neg_six;                                                               \
                    st[t][1] = sp1 * sm1 * sm2 / two;                                                                 \
                    st[t][2] = sp1 * s * sm2 / neg_two;                                                               \
                    st[t][3] = sp1 * s * sm1 / six;                                                                   \
                    included[t] = !threshold || d >= rthres;                                                          \
                }                                                                                                     \
                if (!in_grid) {                                                                                       \
                    status = DC_OUT_OF_GRID;                                                                          \
                    break;                                                                                            \
                }                                                                                                     \
                for (int64_t t = 0; t < n; t++) {                                                                     \
                    if (!included[t]) continue;                                                                       \
                    const int64_t c = c0 + t, k = bins[t];                                                            \
                    const T d = distances[t], oj = b.col_occ[c];                                                      \
                    const T *w = st[t];                                                                               \
                    if (need_xyz && d > 0) {                                                                          \
                        const T *xj = b.col_xyz + 3 * c;                                                              \
                        const T dx = xi - xj[0], dy = yi - xj[1], dz = zi - xj[2];                                    \
                        const T *h = h_table + k;                                                                     \
                        const T hv = w[0] * h[0] + w[1] * h[1] + w[2] * h[2] + w[3] * h[3];                           \
                        const T coef = hv / d * (oi * oj);                                                            \
                        const double cx = (double)(coef * dx), cy = (double)(coef * dy), cz = (double)(coef * dz);    \
                        gx += cx;                                                                                     \
                        gy += cy;                                                                                     \
                        gz += cz;                                                                                     \
                        col_gxyz[3 * c] -= cx;                                                                        \
                        col_gxyz[3 * c + 1] -= cy;                                                                    \
                        col_gxyz[3 * c + 2] -= cz;                                                                    \
                    }                                                                                                 \
                    if (need_occ) {                                                                                   \
                        const T *kk = k_table + k;                                                                    \
                        const T kv = w[0] * kk[0] + w[1] * kk[1] + w[2] * kk[2] + w[3] * kk[3];                       \
                        go += (double)(kv * oj);                                                                      \
                        col_gocc[c] += (double)(kv * oi);                                                             \
                    }                                                                                                 \
                }                                                                                                     \
            }                                                                                                         \
            if (need_xyz) {                                                                                           \
                grad_xyz[3 * i] += gx;                                                                                \
                grad_xyz[3 * i + 1] += gy;                                                                            \
                grad_xyz[3 * i + 2] += gz;                                                                            \
            }                                                                                                         \
            if (need_occ) grad_occ[i] += go;                                                                          \
        }                                                                                                             \
        if (status == DC_OK) {                                                                                        \
            for (int64_t c = 0; c < b.n_cols; c++) {                                                                  \
                const int64_t j = b.cols[c];                                                                          \
                if (need_xyz) {                                                                                       \
                    grad_xyz[3 * j] += col_gxyz[3 * c];                                                               \
                    grad_xyz[3 * j + 1] += col_gxyz[3 * c + 1];                                                       \
                    grad_xyz[3 * j + 2] += col_gxyz[3 * c + 2];                                                       \
                }                                                                                                     \
                if (need_occ) grad_occ[j] += col_gocc[c];                                                             \
            }                                                                                                         \
        }                                                                                                             \
        free(col_gxyz);                                                                                               \
        free(col_gocc);                                                                                               \
        batch_free_##SUFFIX(&b);                                                                                      \
        return status;                                                                                                \
    }

DEFINE_KERNELS(float, f32, sqrtf, floorf)
DEFINE_KERNELS(double, f64, sqrt, floor)
