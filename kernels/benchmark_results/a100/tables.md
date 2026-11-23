
**I(Q), float32**: time (speed-up vs PyTorch CPU) / peak memory (RAM + GPU tensors)

| atoms | Rust CPU | PyTorch CUDA | Triton CUDA |
|---:|---:|---:|---:|
| 473 | 4.7 ms / 2 MB | 10.0 ms / 3 MB | 8.4 ms / 3 MB |
| 3,833 | 71.7 ms / 61 MB | 36.7 ms / 94 MB | 11.3 ms / 6 MB |
| 12,581 | 123.4 ms / 98 MB | 187.0 ms / 122 MB | 26.0 ms / 22 MB |
| 41,801 | 461.4 ms / 158 MB | 1.89 s / 120 MB | 104.6 ms / 62 MB |
| 112,331 | 2.74 s / 292 MB | 13.47 s / 165 MB | 628.5 ms / 213 MB |
| 228,653 | 12.40 s / 620 MB | 56.52 s / 335 MB | 2.37 s / 337 MB |
| 444,263 | 52.15 s / 1129 MB | 217.03 s / 651 MB | 8.61 s / 655 MB |

**I(Q) + gradient, float32**: time (speed-up vs PyTorch CPU) / peak memory (RAM + GPU tensors)

| atoms | Rust CPU | PyTorch CUDA | Triton CUDA |
|---:|---:|---:|---:|
| 473 | 17.3 ms / 48 MB | 40.5 ms / 53 MB | 22.1 ms / 44 MB |
| 3,833 | 87.0 ms / 118 MB | 89.4 ms / 281 MB | 26.7 ms / 129 MB |
| 12,581 | 211.1 ms / 138 MB | 407.8 ms / 247 MB | 41.6 ms / 117 MB |
| 41,801 | 1.13 s / 144 MB | 4.16 s / 289 MB | 161.0 ms / 117 MB |
| 112,331 | 8.54 s / 238 MB | 35.33 s / 264 MB | 1.01 s / 175 MB |
| 228,653 | 52.61 s / 371 MB | 145.54 s / 350 MB | 3.90 s / 334 MB |
| 444,263 | 217.43 s / 820 MB | 569.56 s / 672 MB | 14.45 s / 650 MB |

**I(Q), float64**: time (speed-up vs PyTorch CPU) / peak memory (RAM + GPU tensors)

| atoms | Rust CPU | PyTorch CUDA | Triton CUDA |
|---:|---:|---:|---:|
| 473 | 6.3 ms / 1 MB | 9.3 ms / 41 MB | 8.4 ms / 113 MB |
| 3,833 | 118.3 ms / 196 MB | 38.3 ms / 209 MB | 11.3 ms / 127 MB |
| 12,581 | 195.3 ms / 322 MB | 199.8 ms / 269 MB | 28.0 ms / 138 MB |
| 41,801 | 486.8 ms / 527 MB | 2.02 s / 273 MB | 133.2 ms / 270 MB |
| 112,331 | 3.56 s / 814 MB | 14.50 s / 355 MB | 697.2 ms / 472 MB |
| 228,653 | 17.83 s / 1301 MB | 61.04 s / 689 MB | 2.46 s / 761 MB |
| 444,263 | 73.96 s / 2255 MB | 235.20 s / 1310 MB | 8.81 s / 1384 MB |

**I(Q) + gradient, float64**: time (speed-up vs PyTorch CPU) / peak memory (RAM + GPU tensors)

| atoms | Rust CPU | PyTorch CUDA | Triton CUDA |
|---:|---:|---:|---:|
| 473 | 33.5 ms / 137 MB | 46.2 ms / 116 MB | 27.1 ms / 171 MB |
| 3,833 | 163.3 ms / 258 MB | 90.5 ms / 433 MB | 30.6 ms / 193 MB |
| 12,581 | 309.7 ms / 264 MB | 504.8 ms / 443 MB | 64.2 ms / 144 MB |
| 41,801 | 1.23 s / 283 MB | 5.16 s / 511 MB | 227.9 ms / 168 MB |
| 112,331 | 10.75 s / 402 MB | 35.31 s / 470 MB | 1.37 s / 327 MB |
| 228,653 | 66.49 s / 694 MB | 146.64 s / 665 MB | 5.32 s / 663 MB |
| 444,263 | 275.42 s / 1356 MB | 561.83 s / 1294 MB | 20.01 s / 1291 MB |

**GPU memory** (peak above the level before the run; the driver figure includes the allocator's cached memory)

| atoms | backend | I(Q): GPU tensors / GPU driver | I(Q) + gradient: GPU tensors / GPU driver |
|---:|---|---:|---:|
| 473 | PyTorch CUDA | 3 MB / 2 MB | 40 MB / 44 MB |
| 473 | PyTorch CUDA | 11 MB / 4 MB | 116 MB / 130 MB |
| 473 | Triton CUDA | 3 MB / 2 MB | 40 MB / 42 MB |
| 473 | Triton CUDA | 11 MB / 4 MB | 116 MB / 124 MB |
| 3,833 | PyTorch CUDA | 94 MB / 102 MB | 267 MB / 286 MB |
| 3,833 | PyTorch CUDA | 176 MB / 202 MB | 433 MB / 562 MB |
| 3,833 | Triton CUDA | 6 MB / 2 MB | 81 MB / 82 MB |
| 3,833 | Triton CUDA | 21 MB / 4 MB | 141 MB / 148 MB |
| 12,581 | PyTorch CUDA | 118 MB / 176 MB | 233 MB / 296 MB |
| 12,581 | PyTorch CUDA | 239 MB / 348 MB | 443 MB / 524 MB |
| 12,581 | Triton CUDA | 18 MB / 24 MB | 116 MB / 144 MB |
| 12,581 | Triton CUDA | 36 MB / 42 MB | 141 MB / 188 MB |
| 41,801 | PyTorch CUDA | 119 MB / 134 MB | 275 MB / 318 MB |
| 41,801 | PyTorch CUDA | 243 MB / 262 MB | 511 MB / 566 MB |
| 41,801 | Triton CUDA | 61 MB / 66 MB | 117 MB / 138 MB |
| 41,801 | Triton CUDA | 122 MB / 146 MB | 144 MB / 196 MB |
| 112,331 | PyTorch CUDA | 163 MB / 216 MB | 244 MB / 310 MB |
| 112,331 | PyTorch CUDA | 321 MB / 594 MB | 468 MB / 612 MB |
| 112,331 | Triton CUDA | 163 MB / 168 MB | 164 MB / 168 MB |
| 112,331 | Triton CUDA | 321 MB / 366 MB | 324 MB / 386 MB |
| 228,653 | PyTorch CUDA | 329 MB / 472 MB | 332 MB / 486 MB |
| 228,653 | PyTorch CUDA | 654 MB / 944 MB | 660 MB / 960 MB |
| 228,653 | Triton CUDA | 329 MB / 332 MB | 332 MB / 332 MB |
| 228,653 | Triton CUDA | 654 MB / 720 MB | 660 MB / 740 MB |
| 444,263 | PyTorch CUDA | 639 MB / 780 MB | 644 MB / 798 MB |
| 444,263 | PyTorch CUDA | 1275 MB / 1550 MB | 1285 MB / 1580 MB |
| 444,263 | Triton CUDA | 639 MB / 652 MB | 644 MB / 672 MB |
| 444,263 | Triton CUDA | 1275 MB / 1358 MB | 1285 MB / 1370 MB |

**Accuracy** (relative to the float64 reference)

| atoms | backend | dtype | I(Q) max. rel. error | gradient rel. error |
|---:|---|---|---:|---:|
| 473 | Rust CPU | float32 | 5.3e-07 | 6.0e-07 |
| 473 | Rust CPU | float64 | 0.0e+00 | 0.0e+00 |
| 473 | PyTorch CUDA | float32 | 4.8e-07 | 1.5e-06 |
| 473 | PyTorch CUDA | float64 | 5.7e-16 | 4.8e-15 |
| 473 | Triton CUDA | float32 | 4.3e-07 | 8.6e-07 |
| 473 | Triton CUDA | float64 | 7.6e-16 | 2.1e-15 |
| 3,833 | Rust CPU | float32 | 4.9e-07 | 4.2e-06 |
| 3,833 | Rust CPU | float64 | 0.0e+00 | 0.0e+00 |
| 3,833 | PyTorch CUDA | float32 | 8.4e-07 | 7.8e-06 |
| 3,833 | PyTorch CUDA | float64 | 2.0e-15 | 2.4e-14 |
| 3,833 | Triton CUDA | float32 | 6.6e-07 | 3.3e-06 |
| 3,833 | Triton CUDA | float64 | 3.0e-15 | 3.8e-15 |
| 12,581 | Rust CPU | float32 | 7.3e-07 | 4.5e-05 |
| 12,581 | Rust CPU | float64 | 0.0e+00 | 0.0e+00 |
| 12,581 | PyTorch CUDA | float32 | 9.6e-07 | 4.5e-05 |
| 12,581 | PyTorch CUDA | float64 | 1.2e-15 | 7.5e-15 |
| 12,581 | Triton CUDA | float32 | 1.0e-06 | 2.8e-05 |
| 12,581 | Triton CUDA | float64 | 1.8e-15 | 2.6e-15 |
| 41,801 | Rust CPU | float32 | 1.8e-06 | 4.4e-05 |
| 41,801 | Rust CPU | float64 | 0.0e+00 | 0.0e+00 |
| 41,801 | PyTorch CUDA | float32 | 1.9e-06 | 4.4e-05 |
| 41,801 | PyTorch CUDA | float64 | 7.6e-16 | 8.1e-15 |
| 41,801 | Triton CUDA | float32 | 1.6e-06 | 2.7e-05 |
| 41,801 | Triton CUDA | float64 | 4.3e-15 | 4.6e-15 |
| 112,331 | Rust CPU | float32 | 4.6e-06 | 3.2e-05 |
| 112,331 | Rust CPU | float64 | 0.0e+00 | 0.0e+00 |
| 112,331 | PyTorch CUDA | float32 | 4.6e-06 | 3.3e-05 |
| 112,331 | PyTorch CUDA | float64 | 1.7e-15 | 1.3e-14 |
| 112,331 | Triton CUDA | float32 | 3.5e-06 | 2.7e-05 |
| 112,331 | Triton CUDA | float64 | 1.4e-14 | 8.9e-15 |
| 228,653 | Rust CPU | float32 | 7.3e-06 | 3.0e-05 |
| 228,653 | Rust CPU | float64 | 0.0e+00 | 0.0e+00 |
| 228,653 | PyTorch CUDA | float32 | 7.4e-06 | 3.1e-05 |
| 228,653 | PyTorch CUDA | float64 | 7.8e-15 | 1.7e-14 |
| 228,653 | Triton CUDA | float32 | 4.6e-06 | 3.0e-05 |
| 228,653 | Triton CUDA | float64 | 3.7e-14 | 1.3e-14 |
| 444,263 | Rust CPU | float32 | 9.2e-06 | 3.6e-05 |
| 444,263 | Rust CPU | float64 | 0.0e+00 | 0.0e+00 |
| 444,263 | PyTorch CUDA | float32 | 9.2e-06 | 3.6e-05 |
| 444,263 | PyTorch CUDA | float64 | 5.0e-15 | 2.0e-14 |
| 444,263 | Triton CUDA | float32 | 8.9e-06 | 3.4e-05 |
| 444,263 | Triton CUDA | float64 | 7.5e-14 | 1.5e-14 |
