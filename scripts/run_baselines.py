"""Additional baselines: MF-BPR, SGL, NCL, SimGCL, LightGCL, NT-SSM and Turbo-CF."""
from common import (BASELINES, SELECT, SIMGCL_EXTENDED, done, get_selection, load_json, parser, paths, save_json,
                    set_selection, train_cfg)
from specres.config import free_memory
from specres.data import load
from specres.filters import tune_turbocf
from specres.metrics import evaluate_scores, result_record
from specres.models import scores_of
from specres.training import train


def run_one(args, data, name, cfg, seed):
    if done(args, name, seed):
        return
    model, info = train(data, cfg, seed)
    sf = scores_of(model)
    v = evaluate_scores(sf, data, "val")
    t, pu = evaluate_scores(sf, data, "test", per_user=True)
    save_json(paths(args, name, seed)[0], result_record(name, cfg, v, t, pu, **info))
    print(f"{args.dataset} s{seed} {name}: val R@20={v['Recall@20']:.4f} test N@20={t['NDCG@20']:.4f}", flush=True)
    del model
    free_memory()


def main():
    p = parser(__doc__)
    p.add_argument("--methods", nargs="+", default=list(BASELINES) + ["turbocf"],
                   choices=list(BASELINES) + ["turbocf"])
    p.add_argument("--simgcl-extended", action="store_true",
                   help="add lower contrastive weights and noise 0.2 to the SimGCL grid")
    args = p.parse_args()
    grids = {k: [train_cfg(args, **c) for c in v] for k, v in BASELINES.items()}
    if args.simgcl_extended:
        grids["simgcl"] += [train_cfg(args, **c) for c in SIMGCL_EXTENDED]
    for seed in sorted(args.seeds, key=lambda s: s != 42):
        data = load(args.dataset, seed, args.data_dir)
        for key in args.methods:
            if key == "turbocf":
                if not done(args, "turbocf", seed):
                    f, cfg, v = tune_turbocf(data)
                    t, pu = evaluate_scores(f, data, "test", per_user=True)
                    save_json(paths(args, "turbocf", seed)[0], result_record("turbocf", cfg, v, t, pu))
                    del f
                    free_memory()
                continue
            grid = grids[key]
            sel = get_selection(args).get(key)
            if sel is None or len(grid) > get_selection(args).get(f"{key}_grid", len(grid)):
                if seed != 42:
                    raise SystemExit(f"run seed 42 first: it selects the {key} configuration")
                for i, cfg in enumerate(grid):
                    run_one(args, data, f"{key}_c{i}", cfg, seed)
                sel = max(range(len(grid)), key=lambda i: load_json(paths(args, f"{key}_c{i}", seed)[0])["val_" + SELECT])
                set_selection(args, key, sel)
                set_selection(args, f"{key}_grid", len(grid))
            run_one(args, data, f"{key}_c{sel}", grid[sel], seed)


if __name__ == "__main__":
    main()
