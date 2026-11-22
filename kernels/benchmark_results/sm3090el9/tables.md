
**I(Q), float32**: time (speed-up vs PyTorch CPU) / peak memory (RAM + GPU tensors)

| atoms | Rust CPU | PyTorch CUDA | Triton CUDA |
|---:|---:|---:|---:|
| 473 | 3.5 ms / 0 MB | 4.1 ms / 3 MB | 3.7 ms / 3 MB |
| 3,833 | 28.0 ms / 15 MB | 28.8 ms / 94 MB | 5.6 ms / 6 MB |
| 12,581 | 126.4 ms / 28 MB | 191.9 ms / 118 MB | 17.6 ms / 19 MB |
| 41,801 | 1.19 s / 60 MB | 2.03 s / 120 MB | 100.7 ms / 62 MB |
| 112,331 | 8.76 s / 199 MB | 14.58 s / 165 MB | 606.7 ms / 214 MB |
| 228,653 | 40.08 s / 400 MB | 60.83 s / 333 MB | 2.28 s / 335 MB |
| 444,263 | 182.66 s / 799 MB | 232.34 s / 651 MB | 8.50 s / 653 MB |

**I(Q) + gradient, float32**: time (speed-up vs PyTorch CPU) / peak memory (RAM + GPU tensors)

| atoms | Rust CPU | PyTorch CUDA | Triton CUDA |
|---:|---:|---:|---:|
| 473 | 15.7 ms / 56 MB | 9.4 ms / 41 MB | 6.1 ms / 41 MB |
| 3,833 | 81.3 ms / 127 MB | 54.0 ms / 281 MB | 9.8 ms / 82 MB |
| 12,581 | 377.5 ms / 174 MB | 411.2 ms / 247 MB | 26.3 ms / 117 MB |
| 41,801 | 4.28 s / 108 MB | 4.43 s / 288 MB | 154.6 ms / 117 MB |
| 112,331 | 25.35 s / 183 MB | 32.45 s / 265 MB | 980.3 ms / 177 MB |
| 228,653 | 119.42 s / 336 MB | 135.62 s / 353 MB | 3.97 s / 334 MB |
| 444,263 | 589.32 s / 696 MB | 514.56 s / 669 MB | 14.95 s / 650 MB |

**I(Q), float64**: time (speed-up vs PyTorch CPU) / peak memory (RAM + GPU tensors)

| atoms | Rust CPU | PyTorch CUDA | Triton CUDA |
|---:|---:|---:|---:|
| 473 | 5.2 ms / 1 MB | 5.0 ms / 41 MB | 3.7 ms / 113 MB |
| 3,833 | 34.6 ms / 49 MB | 78.0 ms / 206 MB | 6.2 ms / 123 MB |
| 12,581 | 121.8 ms / 96 MB | 619.7 ms / 268 MB | 20.9 ms / 138 MB |
| 41,801 | 975.6 ms / 192 MB | 6.66 s / 273 MB | 146.4 ms / 271 MB |
| 112,331 | 7.42 s / 422 MB | 47.98 s / 351 MB | 962.2 ms / 471 MB |
| 228,653 | 34.66 s / 813 MB | 198.67 s / 684 MB | 3.83 s / 757 MB |
| 444,263 | 197.45 s / 1550 MB | 758.78 s / 1310 MB | 14.27 s / 1384 MB |

**I(Q) + gradient, float64**: time (speed-up vs PyTorch CPU) / peak memory (RAM + GPU tensors)

| atoms | Rust CPU | PyTorch CUDA | Triton CUDA |
|---:|---:|---:|---:|
| 473 | 40.8 ms / 137 MB | 11.6 ms / 116 MB | 9.0 ms / 119 MB |
| 3,833 | 134.0 ms / 233 MB | 168.3 ms / 434 MB | 20.5 ms / 143 MB |
| 12,581 | 460.6 ms / 384 MB | 1.33 s / 444 MB | 87.0 ms / 144 MB |
| 41,801 | 3.67 s / 271 MB | 14.29 s / 513 MB | 799.0 ms / 168 MB |
| 112,331 | 25.09 s / 339 MB | 102.97 s / 470 MB | 5.61 s / 327 MB |
| 228,653 | 126.54 s / 658 MB | 430.56 s / 666 MB | 22.97 s / 663 MB |
| 444,263 | 725.18 s / 1322 MB | 1648.75 s / 1294 MB | 86.72 s / 1291 MB |

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
| 12,581 | Triton CUDA | 36 MB / 24 MB | 141 MB / 188 MB |
| 41,801 | PyTorch CUDA | 119 MB / 134 MB | 275 MB / 318 MB |
| 41,801 | PyTorch CUDA | 243 MB / 262 MB | 511 MB / 566 MB |
| 41,801 | Triton CUDA | 61 MB / 66 MB | 117 MB / 138 MB |
| 41,801 | Triton CUDA | 122 MB / 146 MB | 144 MB / 196 MB |
| 112,331 | PyTorch CUDA | 163 MB / 296 MB | 244 MB / 310 MB |
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
| 473 | PyTorch CUDA | float32 | 4.8e-07 | 1.6e-06 |
| 473 | PyTorch CUDA | float64 | 7.6e-16 | 4.8e-15 |
| 473 | Triton CUDA | float32 | 4.3e-07 | 8.6e-07 |
| 473 | Triton CUDA | float64 | 9.5e-16 | 2.1e-15 |
| 3,833 | Rust CPU | float32 | 4.9e-07 | 4.2e-06 |
| 3,833 | Rust CPU | float64 | 0.0e+00 | 0.0e+00 |
| 3,833 | PyTorch CUDA | float32 | 8.4e-07 | 7.5e-06 |
| 3,833 | PyTorch CUDA | float64 | 1.7e-15 | 3.6e-14 |
| 3,833 | Triton CUDA | float32 | 6.6e-07 | 3.3e-06 |
| 3,833 | Triton CUDA | float64 | 2.3e-15 | 3.9e-15 |
| 12,581 | Rust CPU | float32 | 7.3e-07 | 4.5e-05 |
| 12,581 | Rust CPU | float64 | 0.0e+00 | 0.0e+00 |
| 12,581 | PyTorch CUDA | float32 | 9.8e-07 | 4.5e-05 |
| 12,581 | PyTorch CUDA | float64 | 8.6e-16 | 7.8e-15 |
| 12,581 | Triton CUDA | float32 | 9.9e-07 | 2.8e-05 |
| 12,581 | Triton CUDA | float64 | 1.3e-15 | 2.6e-15 |
| 41,801 | Rust CPU | float32 | 1.8e-06 | 4.4e-05 |
| 41,801 | Rust CPU | float64 | 0.0e+00 | 0.0e+00 |
| 41,801 | PyTorch CUDA | float32 | 1.9e-06 | 4.4e-05 |
| 41,801 | PyTorch CUDA | float64 | 2.0e-15 | 8.1e-15 |
| 41,801 | Triton CUDA | float32 | 1.6e-06 | 2.7e-05 |
| 41,801 | Triton CUDA | float64 | 4.1e-15 | 4.6e-15 |
| 112,331 | Rust CPU | float32 | 4.6e-06 | 3.2e-05 |
| 112,331 | Rust CPU | float64 | 0.0e+00 | 0.0e+00 |
| 112,331 | PyTorch CUDA | float32 | 4.5e-06 | 3.3e-05 |
| 112,331 | PyTorch CUDA | float64 | 2.8e-15 | 1.2e-14 |
| 112,331 | Triton CUDA | float32 | 3.5e-06 | 2.7e-05 |
| 112,331 | Triton CUDA | float64 | 1.5e-14 | 8.8e-15 |
| 228,653 | Rust CPU | float32 | 7.3e-06 | 3.0e-05 |
| 228,653 | Rust CPU | float64 | 0.0e+00 | 0.0e+00 |
| 228,653 | PyTorch CUDA | float32 | 7.3e-06 | 3.1e-05 |
| 228,653 | PyTorch CUDA | float64 | 6.1e-15 | 1.7e-14 |
| 228,653 | Triton CUDA | float32 | 4.6e-06 | 3.0e-05 |
| 228,653 | Triton CUDA | float64 | 3.8e-14 | 1.3e-14 |
| 444,263 | Rust CPU | float32 | 9.2e-06 | 3.6e-05 |
| 444,263 | Rust CPU | float64 | 0.0e+00 | 0.0e+00 |
| 444,263 | PyTorch CUDA | float32 | 9.2e-06 | 3.6e-05 |
| 444,263 | PyTorch CUDA | float64 | 1.4e-14 | 2.1e-14 |
| 444,263 | Triton CUDA | float32 | 8.9e-06 | 3.4e-05 |
| 444,263 | Triton CUDA | float64 | 7.8e-14 | 1.6e-14 |
