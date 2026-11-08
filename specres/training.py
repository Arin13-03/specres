import time

import torch

from .config import DEVICE, SELECT, set_seed
from .losses import bpr, infonce, kmeans, nt_ssm, proto_nce
from .metrics import evaluate_scores
from .models import build, scores_of


def sample_negatives(bu, pos_keys, n_items):
    bj = torch.randint(0, n_items, bu.shape, device=DEVICE)
    for _ in range(100):
        keys = bu * n_items + bj
        hit = pos_keys[torch.searchsorted(pos_keys, keys).clamp(max=pos_keys.numel() - 1)] == keys
        if not hit.any():
            break
        bj = torch.where(hit, torch.randint(0, n_items, bu.shape, device=DEVICE), bj)
    return bj


def _loss(model, cfg, bu, bi, bj):
    kind = cfg["model"]
    if kind == "ntssm":
        ue, uo, ie, io = model.even_odd()
        reg = model.item_emb.weight[bi].pow(2).sum()
        return nt_ssm(ue[bu], uo[bu], ie[bi], io[bi], cfg), reg
    reg = model.item_emb.weight[bi].pow(2).sum() + model.item_emb.weight[bj].pow(2).sum()
    if kind == "lightgcl":
        eu, ei, gu, gi = model.propagate(train=True)
        uu, ii = torch.unique(bu), torch.unique(bi)
        cl = infonce(gu[uu], eu[uu], cfg["cl_temp"]) + infonce(gi[ii], ei[ii], cfg["cl_temp"])
        return bpr(eu, ei, bu, bi, bj) + cfg["cl_weight"] * cl, reg
    if kind == "xsimgcl":
        eu, ei, lu, li = model.propagate(perturb=True)
        uu, ii = torch.unique(bu), torch.unique(bi)
        k = cfg.get("cl_layer", 1)
        cl = infonce(eu[uu], lu[k][uu], cfg["cl_temp"]) + infonce(ei[ii], li[k][ii], cfg["cl_temp"])
        return bpr(eu, ei, bu, bi, bj) + cfg["cl_weight"] * cl, reg
    if kind == "simgcl":
        eu, ei, _, _ = model.propagate()
        u1, i1, _, _ = model.propagate(perturb=True)
        u2, i2, _, _ = model.propagate(perturb=True)
        uu, ii = torch.unique(bu), torch.unique(bi)
        cl = infonce(u1[uu], u2[uu], cfg["cl_temp"]) + infonce(i1[ii], i2[ii], cfg["cl_temp"])
        return bpr(eu, ei, bu, bi, bj) + cfg["cl_weight"] * cl, reg
    if kind == "sgl":
        eu, ei, _, _ = model.propagate()
        (u1, i1, _, _), (u2, i2, _, _) = [model.propagate(keep=k) for k in model.views]
        uu, ii = torch.unique(bu), torch.unique(bi)
        cl = infonce(u1[uu], u2[uu], cfg["cl_temp"]) + infonce(i1[ii], i2[ii], cfg["cl_temp"])
        return bpr(eu, ei, bu, bi, bj) + cfg["cl_weight"] * cl, reg
    if kind == "ncl":
        eu, ei, lu, li = model.propagate()
        uu, ii = torch.unique(bu), torch.unique(bi)
        h = 2 * cfg.get("hyper_layers", 1)
        loss = bpr(eu, ei, bu, bi, bj)
        loss = loss + cfg["cl_weight"] * (infonce(lu[0][uu], lu[h][uu], cfg["cl_temp"])
                                          + infonce(li[0][ii], li[h][ii], cfg["cl_temp"]))
        (cu, au), (ci, ai) = model.protos
        loss = loss + cfg["proto_weight"] * (proto_nce(lu[0][uu], cu, au[uu], cfg["cl_temp"])
                                             + proto_nce(li[0][ii], ci, ai[ii], cfg["cl_temp"]))
        return loss, reg
    eu, ei, _, _ = model.propagate()
    return bpr(eu, ei, bu, bi, bj), reg


def _epoch_state(model, cfg, data, n_edges):
    if cfg["model"] == "sgl":
        model.views = [torch.rand(n_edges, device=DEVICE) >= cfg["edge_drop"] for _ in range(2)]
    if cfg["model"] == "ncl":
        ku = min(cfg["n_clusters"], max(10, data["n_users"] // 5))
        ki = min(cfg["n_clusters"], max(10, data["n_items"] // 5))
        model.protos = (kmeans(model.user_emb.weight.detach(), ku), kmeans(model.item_emb.weight.detach(), ki))


def train(data, cfg, seed):
    """Trains one model with BPR (or NT-SSM), early stopping on validation; returns the best model."""
    set_seed(seed)
    model = build(data, cfg).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    n_items = data["n_items"]
    tu = torch.tensor(data["train_u"], device=DEVICE)
    ti = torch.tensor(data["train_i"], device=DEVICE)
    pos_keys = torch.sort(tu * n_items + ti).values
    best, best_state, best_epoch, bad, t0 = -1.0, None, 0, 0, time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        _epoch_state(model, cfg, data, tu.shape[0])
        perm = torch.randperm(tu.shape[0], device=DEVICE)
        for s in range(0, tu.shape[0], cfg["batch_size"]):
            idx = perm[s:s + cfg["batch_size"]]
            bu, bi = tu[idx], ti[idx]
            bj = sample_negatives(bu, pos_keys, n_items)
            rec, reg_items = _loss(model, cfg, bu, bi, bj)
            reg = model.user_emb.weight[bu].pow(2).sum() + reg_items
            loss = rec + cfg["l2"] * 0.5 * reg / bu.shape[0]
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch % cfg["eval_every"]:
            continue
        model.eval()
        with torch.no_grad():
            val = evaluate_scores(scores_of(model), data, "val")
        if val[SELECT] > best:
            best, best_epoch, bad = val[SELECT], epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"] and epoch >= cfg["min_epochs"]:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, {"best_epoch": best_epoch, "seconds": time.time() - t0}


def load_model(data, cfg, path):
    model = build(data, cfg).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model
