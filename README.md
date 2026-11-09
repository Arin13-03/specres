# Learnable Low-Rank Spectral Residuals for Graph Collaborative Filtering

SpecRes adds a learned low-rank spectral residual to a trained graph recommender (XSimGCL or LightGCN).
This repository contains the code for all models, baselines, ablations and tables.

## Setup

```bash
git clone <this-repo>
cd specres
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

A CUDA GPU is recommended. The code also runs on Apple Silicon (MPS) and on CPU, but more slowly.
The datasets (LastFM-2K, ML-1M, Personality-2018) are downloaded automatically from GroupLens into `data/` on
first use.

## Repository layout

```
specres/            library code
  data.py           download, filtering and per-user 80/10/10 split
  models.py         LightGCN-style backbone (LightGCN, XSimGCL, SimGCL, SGL, NCL, NT-SSM, MF) and LightGCL
  losses.py         BPR, InfoNCE, NT-SSM, prototype loss
  training.py       training loop with validation early stopping
  filters.py        GF-CF, Turbo-CF and score fusion
  specres.py        spectral basis, SpecRes head and fitting
  metrics.py        full-ranking Recall@K / NDCG@K
scripts/            experiment entry points
run_all.sh          runs every step for all datasets and seeds
```

## Running the experiments

Each script takes `--dataset {lastfm-2k,ml-1m,personality-2018}` and `--seeds` (default `42 123 456`).
Seed 42 is always run first, because it selects each method's configuration on validation data.
Finished runs are skipped when a script is re-run. All results go to `results/<dataset>/`.

**1. Backbones:** GF-CF, LightGCN, XSimGCL (validation-tuned) and the dense XSimGCL+GF-CF fusion.

```bash
python scripts/train_backbones.py --dataset ml-1m
```

**2. Additional baselines:** MF-BPR, SGL, NCL, SimGCL, LightGCL, NT-SSM and Turbo-CF.

```bash
python scripts/run_baselines.py --dataset ml-1m
python scripts/run_baselines.py --dataset personality-2018 --simgcl-extended
```

Use `--simgcl-extended` on Personality-2018, where the default SimGCL settings diverge.
Use `--methods sgl ncl` to run a subset.

**3. SpecRes:** fitted on XSimGCL and on LightGCN, plus the held-out-fit variant and the fine-tuning control.
This step needs step 1.

```bash
python scripts/run_specres.py --dataset ml-1m
```

**4. Analysis:** rank sweep, leave-one-out ablation, learned filters, memory and time.

```bash
python scripts/run_analysis.py --dataset ml-1m --seeds 42
```

**5. Tables and figure.**

```bash
python scripts/make_tables.py
python scripts/plot_filters.py
```

These write `tables/table_main.tex`, `tables/summary.txt` and `tables/fig_filters.pdf`.

**Everything at once:**

```bash
bash run_all.sh
```

**Quick check:** add `--quick` to any script (2 epochs) to verify the pipeline before a full run.

```bash
python scripts/train_backbones.py --dataset lastfm-2k --seeds 42 --quick --out results_quick
```

## Using SpecRes on your own model

```python
from specres import load, spectral_basis, backbone_scores, fit_specres

data = load("ml-1m", seed=42)
base = backbone_scores(model)              # any trained model exposing model.final() -> (user_emb, item_emb)
basis = spectral_basis(data, k=1024)
best = fit_specres(data, base, basis, mode="train", seed=42)
scores = best["score"](user_ids)           # (len(user_ids), n_items)
```
