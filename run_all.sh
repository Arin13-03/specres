#!/usr/bin/env bash
set -euo pipefail
for ds in lastfm-2k ml-1m personality-2018; do
  python scripts/train_backbones.py --dataset "$ds"
  extra=""
  [ "$ds" = "personality-2018" ] && extra="--simgcl-extended"
  python scripts/run_baselines.py --dataset "$ds" $extra
  python scripts/run_specres.py --dataset "$ds"
  python scripts/run_analysis.py --dataset "$ds" --seeds 42
done
python scripts/make_tables.py
python scripts/plot_filters.py
