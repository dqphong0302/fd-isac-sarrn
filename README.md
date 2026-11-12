# fd-isac-sarrn

Code, trained models and result files for the article

> T. K. Q. Nguyen, P. Q. Dang, L. L. Nguyen, T. Doan Van, V. D. Tran,
> **"Pilot-Efficient Semi-Blind Channel Estimation for Full-Duplex ISAC"**, *PLOS ONE* (under revision, PONE-D-26-41608).

The repository reproduces every number, table and figure of the revised manuscript from a fixed set of
random seeds. Running the pipeline below on the authors' machine gives exactly the published CSV files
(bit-for-bit).

## What is in here

| Path | Contents |
|---|---|
| `sarrn/sim.py` | OFDM full-duplex ISAC simulator: multipath communication channel, 3-target radar echoes, self-interference. |
| `sarrn/receiver.py` | 2-iteration IFD-DIC receiver: FDIC, 2-D periodogram, CA-CFAR, echo cancellation, equaliser, and every channel estimator (LS, ridge, LMMSE, RLS, DD-LS, SARRN). |
| `sarrn/model.py` | SARRN network (residual MLP, 4 × 512, LayerNorm, SiLU). |
| `scripts/` | Command-line entry points, one per experiment (table below). |
| `scripts/legacy/` | Port of the original MATLAB dataset/evaluation, kept to show that the Python port matches it. |
| `models/` | Trained networks: `sarrn_da_np1.onnx` (SARRN used in the revision), `sarrn_pilot_anchor_np1.onnx` (ablation), `sarrn_submitted_v36.onnx` (network of the original submission). Training metadata and loss history are next to each file. |
| `results/` | Published result files (per-frame CSV, summary tables with 95 % confidence intervals). |
| `matlab/original/` | MATLAB code of the original submission, unchanged, including the scripts that produced the original Fig 3. |
| `matlab/fixed/` | The four MATLAB files corrected during the revision; each change is marked `% FIX`. |
| `matlab/figures/` | MATLAB scripts that draw Fig 3 (NMSE, one pilot), the BER figure and S2 Fig from `results/figures/nmse_ber_table.csv`; they save editable `.fig` and vector `.eps` files in `results/figures/matlab/`. Fonts, sizes and colours are set in the `SETTINGS` block of each file. Tested with MATLAB R2018a. |
| `original_python/` | Original training / ONNX export scripts of the submission. |
| `run_all.sh` | Runs the full pipeline. |

## Installation

Python 3.12 is used; a CUDA GPU is optional (only training uses it).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing the results

```bash
bash run_all.sh                    # everything (about 2-3 h on an RTX 3060)
bash run_all.sh channel figures    # only some stages, using the released models in models/
```

Stages: `data`, `train`, `channel`, `radar`, `extra`, `figures`. If the `train` stage has not been run,
the evaluation uses the released models in `models/`.

| Manuscript item | Command (run from the repository root) | Output |
|---|---|---|
| Training data (Table 2) | `scripts/make_dataset.py --split train --n 10000 --seed 4001 --np 1` and `--split val --n 2500 --seed 5002` | `data/*.h5` |
| SARRN training | `scripts/train.py --np 1 ... --anchor-key est/ddls/np1 --extra-key est/ridge/np1 --fb 2 --rel-residual --loss logdb --alpha 0` | `runs/sarrn_da/np1/` |
| Fig 3, Fig 6, Table 4, Table 6 | `scripts/eval_channel.py --mc 50 --np 1,2 --methods ...` and `--methods genie` | `results/fig3_final/`, `results/fig3_genie/` |
| Table 5, Fig 4 (radar) | `scripts/eval_radar.py --n 300 --gammas 0,10,20,30 --est ddls --modes fixed,legacy` | `results/radar/` |
| Fig 5, S3 Fig (range-Doppler maps) | `scripts/rdm_figure.py [--legacy-radar]` | `results/rdm/` |
| Radar result per channel estimator (Concern #6) | `scripts/radar_by_estimator.py --n 300 --gammas 0,10,20,30 --nfb 4` then `scripts/radar_by_estimator_summary.py results/radar_by_estimator` | `results/radar_by_estimator/` |
| Table 7 (run time) | `scripts/timing.py --reps 20` | `results/timing.md` |
| Table 8 (model mismatch) | `scripts/eval_mismatch.py --mc 20` | `results/mismatch/` |
| S2 Table (QPSK, 16-QAM) | `scripts/eval_channel.py --mod qpsk|16qam --gamma 0:2:40 ...` then `scripts/modulation_table.py` | `results/mod_*/` |
| Figure files, S1 Table | `scripts/make_figures.py`, `scripts/make_latex_tables.py` | `results/figures/` |
| Editable figures (`.fig`, `.eps`) | in MATLAB: `matlab/figures/plot_fig3_nmse.m`, `plot_fig6_ber.m`, `plot_s2fig_pilot_matched.m` | `results/figures/matlab/` |

The exact arguments are in `run_all.sh`.

### Evaluation protocol

* All estimators are evaluated on the **same frames**: for every SNR point γc and Monte-Carlo index the
  scene (channel, targets, self-interference, data, noise) is drawn from a fixed seed, and every method
  runs inside the same 2-iteration IFD-DIC loop.
* 50 frames per point (1024 subcarriers × 512 symbols, about 5 × 10⁵ data bits per frame).
* NMSE: mean over frames, 95 % bootstrap interval. BER: pooled bit errors, 95 % Clopper-Pearson interval.
* Seeds used for evaluation are disjoint from the training and validation seeds (`train.py` refuses
  overlapping seeds).

### Corrections relative to the original submission

1. **Periodogram** (`calc_periodogram.m`, `receiver.periodogram`): the conjugate of E/F is now taken before the
   FFT. Without it, range peaks fall outside the evaluated window and the sign of the velocity is reversed.
   `--legacy-radar` (or `--modes legacy`) reproduces the original behaviour.
2. **RLS** (`NMSE_RLS.m`): the inverse correlation matrix P is reset at each frame. The submitted behaviour
   is still available as method `rls_carryP`.
3. **NMSE_LS.m**: noise level aligned with the other estimators (0 dB).
4. **NMSE_DL.m**: transpose `h.'` instead of the conjugate transpose `h'`.

## Notes

* `matlab/original/useful_function/` contains helper files (including LDPC routines) taken from earlier
  work of the group; they are provided only to document the original submission.
* Numbers depend on floating-point libraries. Other GPUs or library versions may change the last digits of
  the NMSE values, but not the conclusions.

## Versions

* **v1.1.0** — Names as in the revised paper: the proposed estimator is **SARRN** (DD-LS anchor + ResMLP;
  code key `sarrn_da`, class `SARRN_DA`); the network of the original submission is "ResMLP, original
  submission" (code key `sarrn_v36`). Fig 3 and the BER figure show one-pilot estimators only.
  Added `scripts/radar_by_estimator.py` (radar result per channel estimator on the same 300 random scenes)
  with results in `results/radar_by_estimator/`, MATLAB figure scripts in `matlab/figures/` (editable
  `.fig`/`.eps` in `results/figures/matlab/`), and the range-Doppler figure in `results/rdm/`.
  No simulation result changed: `results/figures/nmse_ber_table.csv` is byte-identical to v1.0.0.
* **v1.0.0** — Code, models and results for the PLOS ONE revision (PONE-D-26-41608).

## Citation

See `CITATION.cff`. The article reference will be added after publication.

## License

MIT (see `LICENSE`).
