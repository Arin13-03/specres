"""Ablations (rank k, leave-one-out), learned filter export, and memory/time measurements (one seed)."""
import time

import torch

from common import get_selection, parser, paths, save_json, train_cfg, xsimgcl_grid
from specres.config import DEVICE, free_memory
from specres.data import load
from specres.filters import GFCF
from specres.metrics import evaluate_scores
from specres.specres import backbone_scores, fit_specres, spectral_basis
from specres.training import load_model


def ndcg(score_fn, data):
    return evaluate_scores(score_fn, data, "test")["NDCG@20"]


def timed(fn, reps=5):
    sync = torch.cuda.synchronize if DEVICE.type == "cuda" else (torch.mps.synchronize if DEVICE.type == "mps" else (lambda: None))
    with torch.no_grad():
        fn()
        sync()
        t0 = time.time()
        for _ in range(reps):
            fn()
        sync()
    return 1000 * (time.time() - t0) / reps


def main():
    p = parser(__doc__)
    p.add_argument("--ranks", type=int, nargs="+", default=[64, 128, 256, 512, 1024])
    args = p.parse_args()
    seed = args.seeds[0]
    sel = get_selection(args)["xsimgcl"]
    cfg = train_cfg(args, **xsimgcl_grid()[sel])
    data = load(args.dataset, seed, args.data_dir)
    model = load_model(data, cfg, paths(args, f"xsimgcl_c{sel}", seed)[1])
    base = backbone_scores(model)
    out = {"seed": seed, "backbone": ndcg(lambda u: base[u], data), "rank": {}}

    for k in args.ranks:
        basis = spectral_basis(data, k)
        r = fit_specres(data, base, basis, "train", seed)
        out["rank"][k] = {"ndcg20": ndcg(r["score"], data), "variant": r["variant"]}
        print(f"{args.dataset} k={k}: {out['rank'][k]}", flush=True)
        if k == max(args.ranks):
            head = r["head"]
            with torch.no_grad():
                out["filter"] = {"variant": r["variant"], "sigma": head.sn.cpu().tolist(),
                                 "response": (head.h * head.log_beta.exp()).cpu().tolist()}
            r_noloo = fit_specres(data, base, basis, "train", seed, loo=False)
            out["no_loo"] = {"ndcg20": ndcg(r_noloo["score"], data), "variant": r_noloo["variant"]}
            eu, ei = model.final()
            u = torch.arange(min(1000, data["n_users"]), device=DEVICE)
            f = GFCF(data, 256)
            nU, nI = data["n_users"], data["n_items"]
            out["cost"] = {
                "users": len(u),
                "backbone_ms": timed(lambda: eu[u] @ ei.t()),
                "specres_ms": timed(lambda: eu[u] @ ei.t() + (head.Z[u] * head.h) @ head.Q.t()),
                "dense_ms": timed(lambda: f(u)),
                "specres_extra_MB": 4 * (nU + nI) * basis[3].shape[1] / 1e6,
                "dense_extra_MB": 4 * nI * nI / 1e6,
            }
            del f
        free_memory()
    save_json(f"{args.out}/{args.dataset}/analysis.json", out)


if __name__ == "__main__":
    main()
