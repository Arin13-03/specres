import torch
import torch.nn as nn
import torch.nn.functional as F


def _edges(data):
    eu = torch.tensor(data["train_u"], dtype=torch.long)
    ei = torch.tensor(data["train_i"], dtype=torch.long)
    du = torch.bincount(eu, minlength=data["n_users"]).float().clamp(min=1)
    di = torch.bincount(ei, minlength=data["n_items"]).float().clamp(min=1)
    return eu, ei, 1.0 / torch.sqrt(du[eu] * di[ei])


class GraphCF(nn.Module):
    """LightGCN-style propagation. kind: lightgcn | xsimgcl | simgcl | sgl | ncl | ntssm (layers=0 gives MF)."""

    def __init__(self, data, cfg):
        super().__init__()
        self.cfg, self.kind, self.L = cfg, cfg["model"], cfg["layers"]
        self.user_emb = nn.Embedding(data["n_users"], cfg["dim"])
        self.item_emb = nn.Embedding(data["n_items"], cfg["dim"])
        init = nn.init.xavier_uniform_ if cfg.get("init") == "xavier" else (lambda w: nn.init.normal_(w, std=0.1))
        init(self.user_emb.weight)
        init(self.item_emb.weight)
        eu, ei, norm = _edges(data)
        self.register_buffer("edge_u", eu)
        self.register_buffer("edge_i", ei)
        self.register_buffer("edge_norm", norm.unsqueeze(1))

    def layers(self, perturb=False, keep=None):
        eu, ei = self.user_emb.weight, self.item_emb.weight
        U, I, norm = self.edge_u, self.edge_i, self.edge_norm
        if keep is not None:
            U, I = U[keep], I[keep]
            du = torch.bincount(U, minlength=eu.shape[0]).float().clamp(min=1)
            di = torch.bincount(I, minlength=ei.shape[0]).float().clamp(min=1)
            norm = (1.0 / torch.sqrt(du[U] * di[I])).unsqueeze(1)
        eps = self.cfg.get("eps", 0.0) if perturb else 0.0
        lu, li = [eu], [ei]
        for _ in range(self.L):
            nu = torch.zeros_like(eu).index_add_(0, U, norm * ei[I])
            ni = torch.zeros_like(ei).index_add_(0, I, norm * eu[U])
            if eps > 0:
                nu = nu + torch.sign(nu) * F.normalize(torch.rand_like(nu), dim=1) * eps
                ni = ni + torch.sign(ni) * F.normalize(torch.rand_like(ni), dim=1) * eps
            eu, ei = nu, ni
            lu.append(eu)
            li.append(ei)
        return lu, li

    def propagate(self, perturb=False, keep=None):
        lu, li = self.layers(perturb, keep)
        start = 1 if self.kind in ("xsimgcl", "simgcl") else 0
        return torch.stack(lu[start:]).mean(0), torch.stack(li[start:]).mean(0), lu, li

    def even_odd(self):
        lu, li = self.layers()
        c = self.L + 1
        return sum(lu[0::2]) / c, sum(lu[1::2]) / c, sum(li[0::2]) / c, sum(li[1::2]) / c

    def final(self):
        if self.kind == "ntssm":
            ue, uo, ie, io = self.even_odd()
            return ue + uo, ie + io
        eu, ei, _, _ = self.propagate()
        return eu, ei


class LightGCL(nn.Module):
    """LightGCN propagation plus an SVD-reconstructed view (rank q) used for contrastive learning."""

    def __init__(self, data, cfg):
        super().__init__()
        self.cfg, self.kind, self.L = cfg, "lightgcl", cfg["layers"]
        self.user_emb = nn.Embedding(data["n_users"], cfg["dim"])
        self.item_emb = nn.Embedding(data["n_items"], cfg["dim"])
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
        eu, ei, norm = _edges(data)
        self.register_buffer("edge_u", eu)
        self.register_buffer("edge_i", ei)
        self.register_buffer("edge_norm", norm.unsqueeze(1))
        A = torch.sparse_coo_tensor(torch.stack([eu, ei]), norm, (data["n_users"], data["n_items"])).coalesce()
        U, S, V = torch.svd_lowrank(A, q=cfg["svd_q"], niter=6)
        self.register_buffer("US", U * S)
        self.register_buffer("VS", V * S)
        self.register_buffer("Ut", U.t().contiguous())
        self.register_buffer("Vt", V.t().contiguous())

    def propagate(self, train=False):
        eu, ei = self.user_emb.weight, self.item_emb.weight
        U, I, norm = self.edge_u, self.edge_i, self.edge_norm
        p = self.cfg.get("edge_drop", 0.0) if train else 0.0
        if p > 0:
            keep = torch.rand(U.shape[0], device=U.device) >= p
            U, I, norm = U[keep], I[keep], norm[keep] / (1 - p)
        su, si, gu, gi = eu, ei, eu, ei
        for _ in range(self.L):
            zu = torch.zeros_like(eu).index_add_(0, U, norm * ei[I])
            zi = torch.zeros_like(ei).index_add_(0, I, norm * eu[U])
            gu = gu + self.US @ (self.Vt @ ei)
            gi = gi + self.VS @ (self.Ut @ eu)
            eu, ei = zu, zi
            su, si = su + eu, si + ei
        return su, si, gu, gi

    def final(self):
        su, si, _, _ = self.propagate()
        return su, si


def build(data, cfg):
    return LightGCL(data, cfg) if cfg["model"] == "lightgcl" else GraphCF(data, cfg)


def scores_of(model):
    eu, ei = model.final()
    return lambda users: eu[users] @ ei.t()
