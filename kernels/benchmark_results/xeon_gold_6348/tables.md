
**I(Q), float32**: time (speed-up vs PyTorch CPU) / peak memory (RAM + GPU tensors)

| atoms | PyTorch CPU (1.1) | Rust CPU | C CPU |
|---:|---:|---:|---:|
| 473 | 3.6 ms / 3 MB | 2.7 ms (1.3×) / 2 MB | 2.6 ms (1.4×) / 2 MB |
| 3,833 | 82.9 ms / 273 MB | 31.8 ms (2.6×) / 105 MB | 35.6 ms (2.3×) / 103 MB |
| 12,581 | 637.9 ms / 438 MB | 69.4 ms (9.2×) / 170 MB | 98.8 ms (6.5×) / 168 MB |
| 41,801 | 8.30 s / 546 MB | 316.3 ms (26.2×) / 277 MB | 707.7 ms (11.7×) / 268 MB |
| 112,331 | 57.14 s / 766 MB | 1.70 s (33.6×) / 468 MB | 5.00 s (11.4×) / 427 MB |
| 228,653 | – | 6.13 s / 805 MB | 11.06 s / 726 MB |
| 444,263 | – | 27.98 s / 1416 MB | 23.94 s / 1266 MB |

**I(Q) + gradient, float32**: time (speed-up vs PyTorch CPU) / peak memory (RAM + GPU tensors)

| atoms | PyTorch CPU (1.1) | Rust CPU | C CPU |
|---:|---:|---:|---:|
| 473 | 13.6 ms / 64 MB | 8.2 ms (1.7×) / 39 MB | 10.2 ms (1.3×) / 33 MB |
| 3,833 | 238.7 ms / 189 MB | 59.4 ms (4.0×) / 118 MB | 52.3 ms (4.6×) / 113 MB |
| 12,581 | 2.43 s / 159 MB | 154.4 ms (15.7×) / 152 MB | 207.1 ms (11.7×) / 119 MB |
| 41,801 | 29.24 s / 242 MB | 578.8 ms (50.5×) / 168 MB | 1.68 s (17.4×) / 185 MB |
| 112,331 | 199.25 s / 529 MB | 6.71 s (29.7×) / 203 MB | 11.52 s (17.3×) / 209 MB |
| 228,653 | – | 30.08 s / 627 MB | 31.07 s / 712 MB |
| 444,263 | – | 129.53 s / 1432 MB | 102.54 s / 1322 MB |

**I(Q), float64**: time (speed-up vs PyTorch CPU) / peak memory (RAM + GPU tensors)

| atoms | PyTorch CPU (1.1) | Rust CPU | C CPU |
|---:|---:|---:|---:|
| 473 | 4.7 ms / 2 MB | 3.4 ms (1.4×) / 1 MB | 3.5 ms (1.4×) / 1 MB |
| 3,833 | 104.9 ms / 806 MB | 71.1 ms (1.5×) / 339 MB | 75.7 ms (1.4×) / 354 MB |
| 12,581 | 868.9 ms / 1245 MB | 118.0 ms (7.4×) / 562 MB | 137.2 ms (6.3×) / 549 MB |
| 41,801 | 9.49 s / 1623 MB | 367.5 ms (25.8×) / 918 MB | 781.2 ms (12.1×) / 897 MB |
| 112,331 | 61.61 s / 2182 MB | 2.07 s (29.8×) / 1421 MB | 5.42 s (11.4×) / 1395 MB |
| 228,653 | – | 9.34 s / 1966 MB | 12.55 s / 1918 MB |
| 444,263 | – | 37.15 s / 2927 MB | 35.32 s / 2827 MB |

**I(Q) + gradient, float64**: time (speed-up vs PyTorch CPU) / peak memory (RAM + GPU tensors)

| atoms | PyTorch CPU (1.1) | Rust CPU | C CPU |
|---:|---:|---:|---:|
| 473 | 24.4 ms / 161 MB | 14.1 ms (1.7×) / 95 MB | 14.3 ms (1.7×) / 74 MB |
| 3,833 | 284.9 ms / 254 MB | 97.8 ms (2.9×) / 265 MB | 101.9 ms (2.8×) / 248 MB |
| 12,581 | 2.77 s / 345 MB | 220.2 ms (12.6×) / 371 MB | 265.0 ms (10.5×) / 358 MB |
| 41,801 | 30.78 s / 1306 MB | 711.4 ms (43.3×) / 495 MB | 2.00 s (15.4×) / 495 MB |
| 112,331 | 206.02 s / 720 MB | 7.89 s (26.1×) / 710 MB | 13.16 s (15.7×) / 703 MB |
| 228,653 | – | 42.98 s / 922 MB | 42.62 s / 909 MB |
| 444,263 | – | 158.37 s / 1371 MB | 145.69 s / 1352 MB |

**Accuracy** (relative to the float64 reference)

| atoms | backend | dtype | I(Q) max. rel. error | gradient rel. error |
|---:|---|---|---:|---:|
| 473 | PyTorch CPU (1.1) | float32 | 5.8e-07 | 3.1e-06 |
| 473 | PyTorch CPU (1.1) | float64 | 0.0e+00 | 4.6e-15 |
| 473 | Rust CPU | float32 | 5.3e-07 | 6.0e-07 |
| 473 | Rust CPU | float64 | 0.0e+00 | 0.0e+00 |
| 473 | C CPU | float32 | 5.3e-07 | 6.0e-07 |
| 473 | C CPU | float64 | 0.0e+00 | 0.0e+00 |
| 3,833 | PyTorch CPU (1.1) | float32 | 6.6e-07 | 9.6e-06 |
| 3,833 | PyTorch CPU (1.1) | float64 | 8.7e-16 | 1.5e-14 |
| 3,833 | Rust CPU | float32 | 4.9e-07 | 4.2e-06 |
| 3,833 | Rust CPU | float64 | 0.0e+00 | 0.0e+00 |
| 3,833 | C CPU | float32 | 4.9e-07 | 4.2e-06 |
| 3,833 | C CPU | float64 | 0.0e+00 | 0.0e+00 |
| 12,581 | PyTorch CPU (1.1) | float32 | 1.0e-06 | 2.8e-05 |
| 12,581 | PyTorch CPU (1.1) | float64 | 6.5e-16 | 4.7e-15 |
| 12,581 | Rust CPU | float32 | 7.3e-07 | 4.5e-05 |
| 12,581 | Rust CPU | float64 | 0.0e+00 | 0.0e+00 |
| 12,581 | C CPU | float32 | 7.3e-07 | 4.5e-05 |
| 12,581 | C CPU | float64 | 0.0e+00 | 0.0e+00 |
| 41,801 | PyTorch CPU (1.1) | float32 | 1.6e-06 | 2.7e-05 |
| 41,801 | PyTorch CPU (1.1) | float64 | 6.7e-16 | 1.8e-14 |
| 41,801 | Rust CPU | float32 | 1.8e-06 | 4.4e-05 |
| 41,801 | Rust CPU | float64 | 0.0e+00 | 0.0e+00 |
| 41,801 | C CPU | float32 | 1.8e-06 | 4.4e-05 |
| 41,801 | C CPU | float64 | 0.0e+00 | 0.0e+00 |
| 112,331 | PyTorch CPU (1.1) | float32 | 3.5e-06 | 4.1e-05 |
| 112,331 | PyTorch CPU (1.1) | float64 | 2.6e-15 | 4.8e-14 |
| 112,331 | Rust CPU | float32 | 4.6e-06 | 3.2e-05 |
| 112,331 | Rust CPU | float64 | 0.0e+00 | 0.0e+00 |
| 112,331 | C CPU | float32 | 4.6e-06 | 3.2e-05 |
| 112,331 | C CPU | float64 | 0.0e+00 | 0.0e+00 |
| 228,653 | Rust CPU | float32 | 7.3e-06 | 3.0e-05 |
| 228,653 | Rust CPU | float64 | 0.0e+00 | 0.0e+00 |
| 228,653 | C CPU | float32 | 7.3e-06 | 3.0e-05 |
| 228,653 | C CPU | float64 | 0.0e+00 | 0.0e+00 |
| 444,263 | Rust CPU | float32 | 9.2e-06 | 3.6e-05 |
| 444,263 | Rust CPU | float64 | 0.0e+00 | 0.0e+00 |
| 444,263 | C CPU | float32 | 9.2e-06 | 3.6e-05 |
| 444,263 | C CPU | float64 | 0.0e+00 | 0.0e+00 |
