import numpy as np
import torch

from .config import DEVICE, K_LIST


@torch.no_grad()
def evaluate_scores(score_fn, data, split, per_user=False):
    """Full-ranking Recall@K / NDCG@K. score_fn maps a user index tensor to (batch, n_items) scores."""
    users = [u for u in sorted(data[split].keys()) if len(data[split][u]) > 0]
    max_k = max(K_LIST)
    disc = 1.0 / torch.log2(torch.arange(2, max_k + 2, device=DEVICE, dtype=torch.float32))
    cum_disc = torch.cumsum(disc, 0)
    sums = {k: np.zeros(2) for k in K_LIST}
    pu_r, pu_n = [], []
    for s in range(0, len(users), 256):
        bu = users[s:s + 256]
        scores = score_fn(torch.tensor(bu, device=DEVICE)).clone()
        mr, mc, gr, gc = [], [], [], []
        for row, u in enumerate(bu):
            masked = list(data["user_pos"][u]) + (data["val"][u] if split == "test" else [])
            mr += [row] * len(masked)
            mc += masked
            gr += [row] * len(data[split][u])
            gc += data[split][u]
        scores[torch.tensor(mr, device=DEVICE), torch.tensor(mc, device=DEVICE)] = -1e9
        gt = torch.zeros_like(scores, dtype=torch.bool)
        gt[torch.tensor(gr, device=DEVICE), torch.tensor(gc, device=DEVICE)] = True
        n_gt = gt.sum(1).float()
        hits = gt.gather(1, scores.topk(max_k, dim=1).indices).float()
        cum = hits.cumsum(1)
        for k in K_LIST:
            rec = cum[:, k - 1] / n_gt
            ndcg = (hits[:, :k] * disc[:k]).sum(1) / cum_disc[(n_gt.clamp(max=k) - 1).long()]
            sums[k] += np.array([rec.sum().item(), ndcg.sum().item()])
            if per_user and k == 20:
                pu_r.append(rec.cpu().numpy())
                pu_n.append(ndcg.cpu().numpy())
    res = {f"{n}@{k}": float(sums[k][j] / len(users)) for k in K_LIST for j, n in enumerate(("Recall", "NDCG"))}
    if per_user:
        return res, {"recall20": np.concatenate(pu_r), "ndcg20": np.concatenate(pu_n)}
    return res


def zrow(s):
    return (s - s.mean(1, keepdim=True)) / s.std(1, keepdim=True).clamp(min=1e-8)


def result_record(name, cfg, val, test, pu, **extra):
    return {"name": name, "cfg": cfg, **{f"val_{k}": v for k, v in val.items()},
            **{f"test_{k}": v for k, v in test.items()},
            "pu_recall20": np.round(pu["recall20"], 6).tolist(), "pu_ndcg20": np.round(pu["ndcg20"], 6).tolist(), **extra}
