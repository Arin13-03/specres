"""SpecRes on XSimGCL and LightGCN, the held-out-fit variant, and the fine-tuning control.
Requires train_backbones.py to have been run for the same dataset and seeds."""
import copy

import torch
import torch.nn.functional as F

from common import done, get_selection, load_json, parser, paths, save_json, train_cfg, xsimgcl_grid, SELECT
from specres.config import DEVICE, free_memory, set_seed
from specres.data import load
from specres.metrics import evaluate_scores, result_record
from specres.models import scores_of
from specres.specres import backbone_scores, fit_specres, spectral_basis, split_validation
from specres.training import load_model, sample_negatives


def finetune_on_heldout(model, data, seed, lrs=(1e-4, 5e-4, 1e-3), epochs=40, patience=4, batch=4096):
    """Control: continue training the backbone with BPR on validation half A, selecting on half B."""
    A, data_b = split_validation(data, seed)
    n_items = data["n_items"]
    pairs = [(u, i) for u, its in A.items() for i in its]
    fu = torch.tensor([p[0] for p in pairs], device=DEVICE)
    fi = torch.tensor([p[1] for p in pairs], device=DEVICE)
    excl = torch.cat([torch.tensor(data["train_u"], device=DEVICE) * n_items
                      + torch.tensor(data["train_i"], device=DEVICE), fu * n_items + fi])
    pos_keys = torch.sort(torch.unique(excl)).values
    best = None
    for lr in lrs:
        set_seed(seed)
        m = copy.deepcopy(model)
        opt = torch.optim.Adam(m.parameters(), lr=lr)
        with torch.no_grad():
            top = evaluate_scores(scores_of(m), data_b, "val")[SELECT]
        state, bad = copy.deepcopy(m.state_dict()), 0
        for _ in range(epochs):
            m.train()
            perm = torch.randperm(fu.shape[0], device=DEVICE)
            for s in range(0, fu.shape[0], batch):
                idx = perm[s:s + batch]
                bu, bi = fu[idx], fi[idx]
                bj = sample_negatives(bu, pos_keys, n_items)
                eu, ei, _, _ = m.propagate()
                loss = -F.logsigmoid((eu[bu] * ei[bi]).sum(1) - (eu[bu] * ei[bj]).sum(1)).mean()
                loss = loss + 1e-4 * 0.5 * (m.user_emb.weight[bu].pow(2).sum() + m.item_emb.weight[bi].pow(2).sum()
                                            + m.item_emb.weight[bj].pow(2).sum()) / bu.shape[0]
                opt.zero_grad()
                loss.backward()
                opt.step()
            m.eval()
            with torch.no_grad():
                v = evaluate_scores(scores_of(m), data_b, "val")[SELECT]
            if v > top:
                top, state, bad = v, copy.deepcopy(m.state_dict()), 0
            else:
                bad += 1
                if bad >= patience:
                    break
        m.load_state_dict(state)
        m.eval()
        if best is None or top > best[1]:
            best = (m, top, lr)
    return best


def evaluate_and_save(args, data, name, score_fn, seed, **extra):
    v = evaluate_scores(score_fn, data, "val")
    t, pu = evaluate_scores(score_fn, data, "test", per_user=True)
    save_json(paths(args, name, seed)[0], result_record(name, extra, v, t, pu))
    print(f"{args.dataset} s{seed} {name}: test N@20={t['NDCG@20']:.4f}", flush=True)


def main():
    p = parser(__doc__)
    p.add_argument("--k", type=int, default=1024)
    args = p.parse_args()
    sel = get_selection(args).get("xsimgcl")
    if sel is None:
        raise SystemExit("run train_backbones.py first")
    xcfg = train_cfg(args, **xsimgcl_grid()[sel])
    backbones = {"xsimgcl": (f"xsimgcl_c{sel}", xcfg), "lightgcn": ("lightgcn", train_cfg(args, model="lightgcn"))}
    for seed in args.seeds:
        data = load(args.dataset, seed, args.data_dir)
        basis = spectral_basis(data, args.k)
        for tag, (name, cfg) in backbones.items():
            model = load_model(data, cfg, paths(args, name, seed)[1])
            base = backbone_scores(model)
            if not done(args, f"specres_{tag}", seed):
                r = fit_specres(data, base, basis, "train", seed)
                evaluate_and_save(args, data, f"specres_{tag}", r["score"], seed, variant=r["variant"], k=args.k)
            if tag == "xsimgcl":
                if not done(args, "specres_heldout", seed):
                    r = fit_specres(data, base, basis, "heldout", seed)
                    evaluate_and_save(args, data, "specres_heldout", r["score"], seed, variant=r["variant"], k=args.k)
                if not done(args, "xsimgcl_finetuned", seed):
                    m, _, lr = finetune_on_heldout(model, data, seed)
                    evaluate_and_save(args, data, "xsimgcl_finetuned", scores_of(m), seed, lr=lr)
            del base, model
            free_memory()


if __name__ == "__main__":
    main()
