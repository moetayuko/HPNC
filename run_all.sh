#!/usr/bin/env bash

# Test for running a single experiment. --repeat means run how many different random seeds.

if [ "$1" = "--ablation" ]; then
  abla_opt="hpnc.rot_centers False hpnc.perform_kmeans True"
else
  abla_opt=
fi

DATASETS="cora"
BACKEND="dec rim"
for data in $DATASETS; do
  for clu in $BACKEND; do
    python main.py --cfg configs/hpnc_$clu/$data.yaml --repeat 5 $abla_opt
  done
done
