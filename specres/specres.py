import random

import torch
import torch.nn.functional as F

from .config import DEVICE, SELECT, set_seed
from .metrics import evaluate_scores
from .training import sample_negatives

VARIANTS = [(g, kind, c) for g in (0.0, 0.5) for kind in ("free", "cutoff") for c in (256, 512)
            if not (g == 0.0 and kind == "free" and c == 512)]


def spectral_basis(data, k=1024):
    """Top-k right singular vectors/values of D_U^-1/2 R D_I^-1/2, plus R and item degrees."""
    nU, nI = data["n_users"], data["n_items"]
    k = min(k, nU, nI)
    R = torch.zeros(nU, nI)
    R[torch.tensor(data["train_u"]), torch.tensor(data["train_i"])] = 1.0
    du, di = R.sum(1).clamp(min=1), R.sum(0).clamp(min=1)
    Rn = R / du.sqrt().unsqueeze(1) / di.sqrt().unsqueeze(0)
    if min(nU, nI) <= 8000:
        _, S, Vh = torch.linalg.svd(Rn, full_matrices=False)
        V, S = Vh.t()[:, :k].contiguous(), S[:k]
    else:
        _, S, V = torch.svd_lowrank(Rn, q=k, niter=6)
    return R, di, S, V


class SpecHead(torch.nn.Module):
    """Learnable low-rank spectral residual: beta * c_u * z_u diag(h) q_i^T."""

    def __init__(self, basis, gamma, kind="free", cut0=256):
        super().__init__()
        R, di, S, V = basis
        cut0 = min(cut0, len(S) - 1)
        dg = di.pow(gamma).unsqueeze(1)
        self.register_buffer("Z", (R @ (V / dg)).to(DEVICE))
        self.register_buffer("Q", (V * dg).to(DEVICE))
        self.register_buffer("V2", V.pow(2).to(DEVICE))
        self.register_buffer("sn", (S / S.max()).to(DEVICE))
        self.register_buffer("s2", (S.pow(2) / S.pow(2).max()).to(DEVICE))
        self.kind = kind
        tau0 = float(S[cut0] / S.max())
        if kind == "free":
            h0 = self.s2.clone() if gamma == 0 else torch.sigmoid((self.sn - tau0) / 0.01)
            self.h_free = torch.nn.Parameter(h0)
        else:
            self.a = torch.nn.Parameter(torch.tensor(0.0 if gamma == 0 else 1.0, device=DEVICE))
            self.b = torch.nn.Parameter(torch.tensor(1.0 if gamma == 0 else 0.0, device=DEVICE))
            self.tau = torch.nn.Parameter(torch.tensor(tau0, device=DEVICE))
            self.log_s = torch.nn.Parameter(torch.tensor(-4.6, device=DEVICE))
        self.log_beta = torch.nn.Parameter(torch.tensor(-1.0, device=DEVICE))
        with torch.no_grad():
            c = torch.cat([self.full(torch.arange(s, min(s + 512, R.shape[0]), device=DEVICE)).std(1)
                           for s in range(0, R.shape[0], 512)])
        self.register_buffer("c", 1.0 / c.clamp(min=1e-8))

    @property
    def h(self):
        if self.kind == "free":
            return self.h_free
        return self.a * torch.sigmoid((self.sn - self.tau) / self.log_s.exp()) + self.b * self.s2

    def full(self, users):
        return (self.Z[users] * self.h) @ self.Q.t()

    def pair(self, users, items, loo=False):
        s = (self.Z[users] * self.h * self.Q[items]).sum(1)
        if loo:
            s = s - (self.V2[items] * self.h).sum(1)
        return s


@torch.no_grad()
def backbone_scores(model):
    eu, ei = model.final()
    S = eu @ ei.t()
    return S / S.std(1, keepdim=True).clamp(min=1e-8)


def split_validation(data, seed):
    rng = random.Random(seed)
    A, B = {}, {}
    for u, items in data["val"].items():
        it = list(items)
        rng.shuffle(it)
        A[u], B[u] = it[:len(it) // 2], it[len(it) // 2:]
    data_b = {**data, "val": B, "user_pos": [set(p) | set(A.get(u, [])) for u, p in enumerate(data["user_pos"])]}
    return A, data_b


def fit_head(data, base, head, mode="train", heldout=None, epochs=60, patience=6, lr=0.01, batch=4096, seed=42, loo=True):
    """mode='train': BPR on training pairs (leave-one-out), early stopping on validation.
    mode='heldout': BPR on validation half A, early stopping on validation half B."""
    set_seed(seed)
    n_items = data["n_items"]
    if mode == "train":
        fu, fi = torch.tensor(data["train_u"], device=DEVICE), torch.tensor(data["train_i"], device=DEVICE)
        sel_data = data
    else:
        A, sel_data = heldout
        pairs = [(u, i) for u, its in A.items() for i in its]
        fu = torch.tensor([p[0] for p in pairs], device=DEVICE)
        fi = torch.tensor([p[1] for p in pairs], device=DEVICE)
        loo = False
    excl = torch.cat([torch.tensor(data["train_u"], device=DEVICE) * n_items
                      + torch.tensor(data["train_i"], device=DEVICE), fu * n_items + fi])
    pos_keys = torch.sort(torch.unique(excl)).values
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    score = lambda u: base[u] + head.log_beta.exp() * head.c[u].unsqueeze(1) * head.full(u)
    with torch.no_grad():
        best = evaluate_scores(score, sel_data, "val")[SELECT]
    best_state, best_ep, bad = {k: v.clone() for k, v in head.state_dict().items()}, 0, 0
    for ep in range(1, epochs + 1):
        perm = torch.randperm(fu.shape[0], device=DEVICE)
        for s in range(0, fu.shape[0], batch):
            idx = perm[s:s + batch]
            bu, bi = fu[idx], fi[idx]
            bj = sample_negatives(bu, pos_keys, n_items)
            beta, cu = head.log_beta.exp(), head.c[bu]
            sp = base[bu, bi] + beta * cu * head.pair(bu, bi, loo=loo)
            sn = base[bu, bj] + beta * cu * head.pair(bu, bj)
            loss = -F.logsigmoid(sp - sn).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            v = evaluate_scores(score, sel_data, "val")[SELECT]
        if v > best:
            best, best_ep, bad = v, ep, 0
            best_state = {k: t.clone() for k, t in head.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    head.load_state_dict(best_state)
    return float(best), best_ep, score


def fit_specres(data, base, basis, mode="train", seed=42, loo=True, variants=VARIANTS):
    """Fits every filter variant and keeps the one with the best validation Recall@20."""
    heldout = split_validation(data, seed) if mode == "heldout" else None
    best = None
    for gamma, kind, cut0 in variants:
        head = SpecHead(basis, gamma, kind, cut0)
        v, ep, score = fit_head(data, base, head, mode, heldout, seed=seed, loo=loo)
        if best is None or v > best["val"]:
            best = {"val": v, "epoch": ep, "variant": f"{kind}/cut0={cut0}/gamma={gamma}", "head": head, "score": score}
    return best
