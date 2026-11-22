
**I(Q), float32**: time (speed-up vs PyTorch CPU) / peak memory (RAM + GPU tensors)

| atoms | PyTorch CPU (1.1) | Rust CPU | C CPU | PyTorch MPS | Metal MPS |
|---:|---:|---:|---:|---:|---:|
| 473 | 4.0 ms / 2 MB | 2.0 ms (2.0×) / 2 MB | 2.1 ms (1.9×) / 1 MB | 26.1 ms (0.2×) / 2 MB | 20.7 ms (0.2×) / 4 MB |
| 3,833 | 22.6 ms / 115 MB | 7.2 ms (3.1×) / 21 MB | 7.4 ms (3.1×) / 20 MB | 63.1 ms (0.4×) / 90 MB | 26.1 ms (0.9×) / 8 MB |
| 12,581 | 176.8 ms / 224 MB | 20.3 ms (8.7×) / 40 MB | 23.8 ms (7.4×) / 45 MB | 200.1 ms (0.9×) / 120 MB | 51.0 ms (3.5×) / 8 MB |
| 41,801 | 1.96 s / 491 MB | 205.8 ms (9.5×) / 105 MB | 239.5 ms (8.2×) / 100 MB | 1.53 s (1.3×) / 340 MB | 291.0 ms (6.7×) / 12 MB |
| 112,331 | 16.37 s / 1760 MB | 1.90 s (8.6×) / 254 MB | 1.98 s (8.3×) / 243 MB | 13.34 s (1.2×) / 1982 MB | 1.65 s (9.9×) / 15 MB |

**I(Q) + gradient, float32**: time (speed-up vs PyTorch CPU) / peak memory (RAM + GPU tensors)

| atoms | PyTorch CPU (1.1) | Rust CPU | C CPU | PyTorch MPS | Metal MPS |
|---:|---:|---:|---:|---:|---:|
| 473 | 12.4 ms / 41 MB | 5.0 ms (2.5×) / 41 MB | 5.2 ms (2.4×) / 40 MB | 55.2 ms (0.2×) / 31 MB | 30.2 ms (0.4×) / 26 MB |
| 3,833 | 73.7 ms / 134 MB | 16.0 ms (4.6×) / 83 MB | 17.6 ms (4.2×) / 83 MB | 120.6 ms (0.6×) / 231 MB | 38.8 ms (1.9×) / 45 MB |
| 12,581 | 677.6 ms / 192 MB | 59.8 ms (11.3×) / 118 MB | 63.9 ms (10.6×) / 116 MB | 606.6 ms (1.1×) / 310 MB | 89.0 ms (7.6×) / 76 MB |
| 41,801 | 7.59 s / 402 MB | 631.6 ms (12.0×) / 102 MB | 710.2 ms (10.7×) / 112 MB | 5.99 s (1.3×) / 1311 MB | 376.8 ms (20.1×) / 64 MB |
| 112,331 | 60.60 s / 1329 MB | 6.54 s (9.3×) / 184 MB | 6.07 s (10.0×) / 128 MB | 80.29 s (0.8×) / 8244 MB | 2.17 s (27.9×) / 64 MB |

**I(Q), float64**: time (speed-up vs PyTorch CPU) / peak memory (RAM + GPU tensors)

| atoms | PyTorch CPU (1.1) | Rust CPU | C CPU |
|---:|---:|---:|---:|
| 473 | 5.7 ms / 11 MB | 4.0 ms (1.4×) / 4 MB | 3.8 ms (1.5×) / 4 MB |
| 3,833 | 25.3 ms / 254 MB | 11.4 ms (2.2×) / 69 MB | 11.7 ms (2.2×) / 91 MB |
| 12,581 | 203.1 ms / 544 MB | 25.9 ms (7.8×) / 160 MB | 29.3 ms (6.9×) / 167 MB |
| 41,801 | 2.24 s / 891 MB | 177.0 ms (12.7×) / 206 MB | 239.6 ms (9.4×) / 210 MB |
| 112,331 | 19.20 s / 2076 MB | 1.63 s (11.7×) / 633 MB | 2.06 s (9.3×) / 613 MB |

**I(Q) + gradient, float64**: time (speed-up vs PyTorch CPU) / peak memory (RAM + GPU tensors)

| atoms | PyTorch CPU (1.1) | Rust CPU | C CPU |
|---:|---:|---:|---:|
| 473 | 23.4 ms / 164 MB | 15.7 ms (1.5×) / 161 MB | 15.8 ms (1.5×) / 160 MB |
| 3,833 | 104.5 ms / 190 MB | 38.9 ms (2.7×) / 138 MB | 39.6 ms (2.6×) / 143 MB |
| 12,581 | 869.4 ms / 247 MB | 95.0 ms (9.1×) / 151 MB | 97.5 ms (8.9×) / 144 MB |
| 41,801 | 9.34 s / 382 MB | 636.9 ms (14.7×) / 144 MB | 732.7 ms (12.7×) / 139 MB |
| 112,331 | 76.37 s / 1033 MB | 6.04 s (12.6×) / 337 MB | 7.56 s (10.1×) / 294 MB |

**GPU memory** (peak above the level before the run; the driver figure includes the allocator's cached memory)

| atoms | backend | I(Q): GPU tensors / GPU driver | I(Q) + gradient: GPU tensors / GPU driver |
|---:|---|---:|---:|
| 473 | PyTorch MPS | 2 MB / 40 MB | 21 MB / 48 MB |
| 473 | Metal MPS | 1 MB / 1 MB | 21 MB / 32 MB |
| 3,833 | PyTorch MPS | 86 MB / 1024 MB | 217 MB / 1024 MB |
| 3,833 | Metal MPS | 3 MB / 33 MB | 40 MB / 64 MB |
| 12,581 | PyTorch MPS | 101 MB / 2056 MB | 223 MB / 2088 MB |
| 12,581 | Metal MPS | 4 MB / 34 MB | 58 MB / 1056 MB |
| 41,801 | PyTorch MPS | 94 MB / 3248 MB | 251 MB / 3705 MB |
| 41,801 | Metal MPS | 6 MB / 1056 MB | 60 MB / 1056 MB |
| 112,331 | PyTorch MPS | 131 MB / 4128 MB | 238 MB / 8393 MB |
| 112,331 | Metal MPS | 9 MB / 1024 MB | 61 MB / 1032 MB |

**Accuracy** (relative to the float64 reference)

| atoms | backend | dtype | I(Q) max. rel. error | gradient rel. error |
|---:|---|---|---:|---:|
| 473 | PyTorch CPU (1.1) | float32 | 5.3e-07 | 2.3e-06 |
| 473 | PyTorch CPU (1.1) | float64 | 0.0e+00 | 5.6e-15 |
| 473 | Rust CPU | float32 | 5.3e-07 | 5.8e-07 |
| 473 | Rust CPU | float64 | 0.0e+00 | 0.0e+00 |
| 473 | C CPU | float32 | 5.3e-07 | 5.8e-07 |
| 473 | C CPU | float64 | 0.0e+00 | 0.0e+00 |
| 473 | PyTorch MPS | float32 | 1.3e-06 | 2.6e-06 |
| 473 | Metal MPS | float32 | 1.3e-06 | 2.3e-06 |
| 3,833 | PyTorch CPU (1.1) | float32 | 4.9e-07 | 1.5e-05 |
| 3,833 | PyTorch CPU (1.1) | float64 | 7.9e-16 | 2.7e-14 |
| 3,833 | Rust CPU | float32 | 4.9e-07 | 4.1e-06 |
| 3,833 | Rust CPU | float64 | 0.0e+00 | 0.0e+00 |
| 3,833 | C CPU | float32 | 4.9e-07 | 4.1e-06 |
| 3,833 | C CPU | float64 | 0.0e+00 | 0.0e+00 |
| 3,833 | PyTorch MPS | float32 | 5.2e-06 | 4.0e-05 |
| 3,833 | Metal MPS | float32 | 4.9e-06 | 4.0e-05 |
| 12,581 | PyTorch CPU (1.1) | float32 | 7.3e-07 | 4.6e-05 |
| 12,581 | PyTorch CPU (1.1) | float64 | 5.7e-16 | 1.3e-14 |
| 12,581 | Rust CPU | float32 | 7.3e-07 | 4.5e-05 |
| 12,581 | Rust CPU | float64 | 0.0e+00 | 0.0e+00 |
| 12,581 | C CPU | float32 | 7.3e-07 | 4.5e-05 |
| 12,581 | C CPU | float64 | 0.0e+00 | 0.0e+00 |
| 12,581 | PyTorch MPS | float32 | 1.3e-05 | 1.3e-05 |
| 12,581 | Metal MPS | float32 | 1.2e-05 | 1.3e-05 |
| 41,801 | PyTorch CPU (1.1) | float32 | 1.8e-06 | 4.6e-05 |
| 41,801 | PyTorch CPU (1.1) | float64 | 1.9e-15 | 2.2e-14 |
| 41,801 | Rust CPU | float32 | 1.8e-06 | 4.3e-05 |
| 41,801 | Rust CPU | float64 | 0.0e+00 | 0.0e+00 |
| 41,801 | C CPU | float32 | 1.8e-06 | 4.3e-05 |
| 41,801 | C CPU | float64 | 0.0e+00 | 0.0e+00 |
| 41,801 | PyTorch MPS | float32 | 1.3e-05 | 2.9e-05 |
| 41,801 | Metal MPS | float32 | 1.2e-05 | 2.9e-05 |
| 112,331 | PyTorch CPU (1.1) | float32 | 4.6e-06 | 4.5e-05 |
| 112,331 | PyTorch CPU (1.1) | float64 | 2.7e-15 | 4.9e-14 |
| 112,331 | Rust CPU | float32 | 4.6e-06 | 3.2e-05 |
| 112,331 | Rust CPU | float64 | 0.0e+00 | 0.0e+00 |
| 112,331 | C CPU | float32 | 4.6e-06 | 3.2e-05 |
| 112,331 | C CPU | float64 | 0.0e+00 | 0.0e+00 |
| 112,331 | PyTorch MPS | float32 | 2.6e-05 | 3.0e-05 |
| 112,331 | Metal MPS | float32 | 2.6e-05 | 3.0e-05 |
