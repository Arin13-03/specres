"""GF-CF, LightGCN and XSimGCL (validation-tuned on seed 42), plus the dense XSimGCL+GF-CF fusion."""
import os

import torch

from common import SELECT, done, get_selection, load_json, parser, paths, save_json, set_selection, train_cfg, xsimgcl_grid
from specres.config import free_memory
from specres.data import load
from specres.filters import best_fusion, tune_gfcf
from specres.metrics import evaluate_scores, result_record
from specres.models import scores_of
from specres.training import load_model, train


def fit_and_save(args, data, name, cfg, seed):
    if done(args, name, seed):
        return
    model, info = train(data, cfg, seed)
    jp, mp = paths(args, name, seed)
    os.makedirs(os.path.dirname(mp), exist_ok=True)
    torch.save(model.state_dict(), mp)
    sf = scores_of(model)
    v = evaluate_scores(sf, data, "val")
    t, pu = evaluate_scores(sf, data, "test", per_user=True)
    save_json(jp, result_record(name, cfg, v, t, pu, **info))
    print(f"{args.dataset} s{seed} {name}: val R@20={v['Recall@20']:.4f} test N@20={t['NDCG@20']:.4f}", flush=True)
    free_memory()


def main():
    args = parser(__doc__).parse_args()
    grid = [train_cfg(args, **c) for c in xsimgcl_grid()]
    for seed in sorted(args.seeds, key=lambda s: s != 42):
        data = load(args.dataset, seed, args.data_dir)
        filt, gcfg, gv = tune_gfcf(data)
        if not done(args, "gfcf", seed):
            t, pu = evaluate_scores(filt, data, "test", per_user=True)
            save_json(paths(args, "gfcf", seed)[0], result_record("gfcf", gcfg, gv, t, pu))
        fit_and_save(args, data, "lightgcn", train_cfg(args, model="lightgcn"), seed)
        sel = get_selection(args).get("xsimgcl")
        if sel is None:
            if seed != 42:
                raise SystemExit("run seed 42 first: it selects the XSimGCL configuration")
            for i, cfg in enumerate(grid):
                fit_and_save(args, data, f"xsimgcl_c{i}", cfg, seed)
            vals = [paths(args, f"xsimgcl_c{i}", seed)[0] for i in range(len(grid))]
            sel = max(range(len(grid)), key=lambda i: load_json(vals[i])["val_" + SELECT])
            set_selection(args, "xsimgcl", sel)
        name = f"xsimgcl_c{sel}"
        fit_and_save(args, data, name, grid[sel], seed)
        if not done(args, "fusion", seed):
            model = load_model(data, grid[sel], paths(args, name, seed)[1])
            v, w, fn = best_fusion(data, scores_of(model), filt)
            t, pu = evaluate_scores(fn, data, "test", per_user=True)
            save_json(paths(args, "fusion", seed)[0], result_record("fusion", {"w": w, "backbone": name}, v, t, pu))
        del filt
        free_memory()


if __name__ == "__main__":
    main()
