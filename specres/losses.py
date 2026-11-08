import torch
import torch.nn.functional as F


def bpr(eu, ei, bu, bi, bj):
    return -F.logsigmoid((eu[bu] * ei[bi]).sum(1) - (eu[bu] * ei[bj]).sum(1)).mean()


def infonce(a, b, temp):
    a, b = F.normalize(a, dim=1), F.normalize(b, dim=1)
    return F.cross_entropy(a @ b.t() / temp, torch.arange(a.shape[0], device=a.device))


def _nt_ssm_side(ue, uo, ie, io, a_u, a_i, tau):
    B = ue.shape[0]
    den = (ue + uo).norm(dim=1).unsqueeze(1) * (ie + io).norm(dim=1).unsqueeze(0) + 1e-8
    c_uu, c_ii = (ue @ io.t()) / den, (uo @ ie.t()) / den
    c_ui, c_iu = (ue @ ie.t()) / den, (uo @ io.t()) / den
    pos = torch.eye(B, device=ue.device)
    r = (c_uu + c_ii + c_ui + c_iu) * pos + (a_u * c_uu + a_i * c_ii + a_i * c_ui + a_u * c_iu) * (1 - pos)
    logits = r / tau
    logits = logits - logits.max(dim=1, keepdim=True)[0]
    return -torch.mean(torch.diag(logits) - torch.logsumexp(logits, dim=1))


def nt_ssm(ue, uo, ie, io, cfg):
    """NT-SSM loss (Lee et al., ICML 2026) with in-batch negatives, both directions."""
    l1 = _nt_ssm_side(ue, uo, ie, io, cfg["a_iu"], cfg["a_ii"], cfg["temp"])
    l2 = _nt_ssm_side(ie, io, ue, uo, cfg["a_uu"], cfg["a_ui"], cfg["temp"])
    return (l1 + l2) / 2


@torch.no_grad()
def kmeans(x, k, iters=10):
    c = x[torch.randperm(x.shape[0], device=x.device)[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(x, c).argmin(1)
        c = torch.zeros_like(c).index_add_(0, a, x) / torch.bincount(a, minlength=k).clamp(min=1).unsqueeze(1).float()
    return F.normalize(c, dim=1), torch.cdist(x, c).argmin(1)


def proto_nce(z, centroids, assign, temp):
    return F.cross_entropy(F.normalize(z, dim=1) @ centroids.t() / temp, assign)
