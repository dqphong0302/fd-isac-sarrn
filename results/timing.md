| Stage (one GSFDIC iteration) | Device | Median time (ms) |
|---|---|---|
| FDIC (SI suppression) | cuda | 1.244 |
| Periodogram (2-D FFT, 32x zero-pad, crop) | cuda | 29.093 |
| Target echo reconstruction (3 targets) | cuda | 2.548 |
| Spectral features (202-dim) | cuda | 0.804 |
| OFDM demod. of E (IFFT) | cuda | 0.297 |
| FDIC (SI suppression) | cpu | 30.530 |
| Periodogram (2-D FFT, 32x zero-pad, crop) | cpu | 614.786 |
| Target echo reconstruction (3 targets) | cpu | 39.723 |
| Spectral features (202-dim) | cpu | 13.166 |
| OFDM demod. of E (IFFT) | cpu | 0.862 |
| CA-CFAR on 410x410 map | cpu | 6.575 |
| Ridge/LS anchor, 1 pilot (1024 samples) | cpu | 0.039 |
| DD-LS on full frame (524,288 samples) | cpu | 32.262 |
| Equalisation (IIR 1/h) of full frame | cpu | 6.982 |
| SARRN ResMLP inference (ONNX, batch 1) | cpu-1thr | 0.065 |

| Item | Operation count |
|---|---|
| ResMLP | 0.90 M real MAC |
| DD-LS | Gram L^2 K + L K = 10.5 M complex MAC, 4x4 solve negligible |
| Ridge 1 pilot | 20.5 k complex MAC |
| Periodogram | Nc x 16384-pt FFTs + 410 x 32768-pt FFTs ≈ 2.18 GFLOP |
| Frame duration | 512 x 12.8 us = 6.55 ms |
