
### Np = 1  (mean NMSE dB / mean BER over 50 frames)
| method | 0 | 10 | 20 | 30 | 40 |
|---|---|---|---|---|---|
| Perfect CSI (genie) | -inf / 1.8e-01 | -inf / 1.2e-02 | -inf / 1.5e-07 | -inf / 0.0e+00 | -inf / 0.0e+00 |
| LS | -20.7 / 1.8e-01 | -29.7 / 1.2e-02 | -41.6 / 1.5e-07 | -50.9 / 0.0e+00 | -56.1 / 0.0e+00 |
| DD-LS (pilot + decided data, no NN) | -18.6 / 1.8e-01 | -36.8 / 1.2e-02 | -70.8 / 1.1e-07 | -81.0 / 0.0e+00 | -89.6 / 0.0e+00 |
| SARRN (proposed: DD-LS anchor + ResMLP) | -18.6 / 1.8e-01 | -37.4 / 1.2e-02 | -70.7 / 1.1e-07 | -81.1 / 0.0e+00 | -89.4 / 0.0e+00 |

### Np = 2  (mean NMSE dB / mean BER over 50 frames)
| method | 0 | 10 | 20 | 30 | 40 |
|---|---|---|---|---|---|
| Perfect CSI (genie) | -inf / 1.8e-01 | -inf / 1.2e-02 | -inf / 1.5e-07 | -inf / 0.0e+00 | -inf / 0.0e+00 |
| LS | -24.8 / 1.8e-01 | -33.7 / 1.2e-02 | -45.7 / 1.3e-07 | -54.8 / 0.0e+00 | -62.3 / 0.0e+00 |
| DD-LS (pilot + decided data, no NN) | -19.2 / 1.8e-01 | -37.5 / 1.2e-02 | -70.7 / 1.1e-07 | -81.0 / 0.0e+00 | -89.6 / 0.0e+00 |
