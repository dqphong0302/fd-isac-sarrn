| Estimator | Trainable parameters | Real MAC per frame | Input preparation (ms) | Network inference (ms) |
|---|---|---|---|---|
| Ridge (no NN), 1 pilot | 0 | – | 0.031 (4-tap ridge fit) | – |
| SARRN (ResMLP, as submitted) | 904,200 | 0.90 M | 0.031 + spectral features (see timing.md) | 0.061 |
| ChannelNet (SRCNN + DnCNN) | 227,970 | 459 M | 0.446 (64-tap LS + FFT) | 9.511 |

CPU, one thread, batch 1, median of 2000 calls (ChannelNet: 200).
