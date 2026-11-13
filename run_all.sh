#!/usr/bin/env bash
# Reproduce every result of the revised manuscript (PLOS ONE, PONE-D-26-41608).
# Usage: bash run_all.sh [stage ...]   stages: data train channel radar extra figures channelnet   (default: all)
# GPU recommended (RTX 3060: about 2-3 h in total). Set PY to choose the interpreter.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
STAGES=${*:-data train channel radar extra figures channelnet}
D=data; R=runs; mkdir -p $D $R results
has() { [[ " $STAGES " == *" $1 "* ]]; }

if has data; then   # training / validation data from the 2-iteration IFD-DIC chain
  [ -f $D/train_np1.h5 ] || $PY scripts/make_dataset.py --split train --n 10000 --seed 4001 --np 1 --out $D/train_np1.h5
  [ -f $D/val_np1.h5 ]   || $PY scripts/make_dataset.py --split val   --n 2500  --seed 5002 --np 1 --out $D/val_np1.h5
fi
if has train; then  # SARRN (anchor = DD-LS) and the pilot-anchor ablation model
  $PY scripts/train.py --np 1 --train $D/train_np1.h5 --val $D/val_np1.h5 --out $R/pilot_anchor/np1
  $PY scripts/train.py --np 1 --train $D/train_np1.h5 --val $D/val_np1.h5 --out $R/sarrn_da/np1 \
      --extra-key est/ridge/np1 --anchor-key est/ddls/np1 --fb 2 --rel-residual --loss logdb --alpha 0
fi
# Evaluation uses freshly trained models if present, otherwise the released ones in models/.
mkdir -p $R/eval/da/np1 $R/eval/pa/np1
DA=$R/sarrn_da/np1/sarrn.onnx;     [ -f $DA ] || DA=models/sarrn_da_np1.onnx
PA=$R/pilot_anchor/np1/sarrn.onnx; [ -f $PA ] || PA=models/sarrn_pilot_anchor_np1.onnx
cp $DA $R/eval/da/np1/sarrn.onnx; cp $PA $R/eval/pa/np1/sarrn.onnx
echo "SARRN model: $DA"
if has channel; then  # Fig 3, Fig 6, Tables 4 and 6 (50 frames per point, same frames for every estimator)
  $PY scripts/eval_channel.py --mc 50 --np 1,2 --etilde-runs $R/eval/pa --da-runs $R/eval/da \
      --methods ls,ridge,lmmse,rls,rls_carryP,sarrn_v36,sarrn_v36_conj,sarrn_etilde,ddls,sarrn_da --out results/fig3_final
  $PY scripts/eval_channel.py --mc 50 --np 1 --methods genie --out results/fig3_genie
fi
if has radar; then    # Table 5, Fig 4, Fig 5, S3 Fig
  $PY scripts/eval_radar.py --n 300 --gammas 0,10,20,30 --est ddls --modes fixed,legacy --out results/radar
  $PY scripts/rdm_figure.py --da-onnx $DA --out results/rdm
  $PY scripts/rdm_figure.py --da-onnx $DA --legacy-radar --out results/rdm
fi
if has extra; then    # Table 7 (timing), S2 Table (QPSK / 16-QAM)
  $PY scripts/timing.py --reps 20 --onnx $DA --out results/timing.md
  for m in qpsk 16qam; do
    $PY scripts/eval_channel.py --mod $m --gamma 0:2:40 --mc 50 --np 1,2 --da-runs $R/eval/da \
        --methods genie,ls,ddls,sarrn_da --out results/mod_$m
  done
  $PY scripts/modulation_table.py --results results
fi
if has figures; then
  $PY scripts/make_figures.py --frames results/fig3_final/frames.csv,results/fig3_genie/frames.csv \
      --radar results/radar/radar_summary.csv --out results/figures
fi
if has channelnet; then  # second revision round: ChannelNet baseline, pilot-matched Fig 3, mismatch, complexity
  if [ ! -f models/channelnet_np1.pt ] || has train; then
    [ -f $D/cn_train_np1.h5 ] || $PY scripts/make_dataset_channelnet.py --split train --n 10000 --seed 4001 --np 1 --out $D/cn_train_np1.h5
    [ -f $D/cn_val_np1.h5 ]   || $PY scripts/make_dataset_channelnet.py --split val   --n 2500  --seed 5002 --np 1 --out $D/cn_val_np1.h5
    $PY scripts/train_channelnet.py --train $D/cn_train_np1.h5 --val $D/cn_val_np1.h5 --out $R/channelnet/np1
  fi
  CNP=$R/channelnet/np1/channelnet.pt; [ -f $CNP ] || CNP=models/channelnet_np1.pt
  mkdir -p $R/eval/cn/np1; cp $CNP $R/eval/cn/np1/channelnet.pt
  $PY scripts/eval_channel.py --mc 50 --np 1 --methods ls,ridge,lmmse,rls,sarrn_v36,channelnet \
      --channelnet $R/eval/cn --out results/fig3_major4
  $PY scripts/make_figures.py --frames results/fig3_major4/frames.csv --radar none --out results/fig3_major4/table
  $PY scripts/eval_mismatch.py --mc 20 --channelnet $CNP --out results/mismatch_v36
  $PY scripts/complexity.py --channelnet $CNP --out results/complexity.md
fi
echo DONE
