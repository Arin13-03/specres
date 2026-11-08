import torch

from .config import DEVICE, SELECT, free_memory
from .metrics import evaluate_scores, zrow


def _rating_matrix(data):
    R = torch.zeros(data["n_users"], data["n_items"], device=DEVICE)
    R[torch.tensor(data["train_u"], device=DEVICE), torch.tensor(data["train_i"], device=DEVICE)] = 1.0
    return R


class GFCF:
    """GF-CF (Shen et al., CIKM 2021): linear low-pass filter plus ideal low-pass filter on the item graph."""

    def __init__(self, data, k=256):
        R = _rating_matrix(data)
        du, di = R.sum(1).clamp(min=1), R.sum(0).clamp(min=1)
        Rn = R / du.sqrt().unsqueeze(1) / di.sqrt().unsqueeze(0)
        _, _, V = torch.svd_lowrank(Rn.cpu(), q=k, niter=4)
        self.R, self.P, self.V, self.sdi = R, Rn.t() @ Rn, V.to(DEVICE), di.sqrt().unsqueeze(1)
        self.set(k, 0.0)

    def set(self, k, alpha):
        self.alpha = alpha
        self.V1, self.V2 = self.V[:, :k] / self.sdi, self.V[:, :k] * self.sdi
        return self

    def __call__(self, users):
        r = self.R[users]
        s = r @ self.P
        if self.alpha > 0:
            s = s + self.alpha * (r @ self.V1) @ self.V2.t()
        return s


def tune_gfcf(data):
    f = GFCF(data, 256)
    best = None
    for k in (64, 128, 256):
        for alpha in (0.0, 0.1, 0.3, 0.5, 1.0):
            if alpha == 0 and k != 64:
                continue
            v = evaluate_scores(f.set(k, alpha), data, "val")
            if best is None or v[SELECT] > best[0][SELECT]:
                best = (v, k, alpha)
    v, k, alpha = best
    return f.set(k, alpha), {"k": k, "alpha": alpha}, v


class TurboCF:
    """Turbo-CF (Park et al., SIGIR 2024): polynomial filter of the element-wise powered item graph."""

    def __init__(self, data, alpha, s, order=1):
        R = _rating_matrix(data)
        du, di = R.sum(1).clamp(min=1), R.sum(0).clamp(min=1)
        Rt = R * du.pow(-alpha).unsqueeze(1) * di.pow(alpha - 1).unsqueeze(0)
        P = Rt.t() @ Rt
        del Rt
        if s != 1.0:
            P.pow_(s)
        x = torch.rand(P.shape[0], 1, device=DEVICE)
        for _ in range(30):
            x = P @ x
            x = x / x.norm()
        P.div_(float((x.t() @ P @ x).item()))
        self.R, self.P, self.order = R, P, order

    def __call__(self, users):
        y = self.R[users] @ self.P
        return 2 * y - y @ self.P if self.order == 2 else y


def tune_turbocf(data):
    best = None
    for alpha in (0.3, 0.5, 0.7):
        for s in (0.6, 0.8, 1.0):
            f = TurboCF(data, alpha, s)
            for order in (1, 2):
                f.order = order
                v = evaluate_scores(f, data, "val")
                if best is None or v[SELECT] > best[0][SELECT]:
                    best = (v, alpha, s, order)
            del f
            free_memory()
    v, alpha, s, order = best
    return TurboCF(data, alpha, s, order), {"alpha": alpha, "s": s, "order": order}, v


def best_fusion(data, score_fn, filt, weights=(0.05, 0.1, 0.2, 0.3, 0.5)):
    best = None
    for w in weights:
        fn = lambda u, w=w: (1 - w) * zrow(score_fn(u)) + w * zrow(filt(u))
        v = evaluate_scores(fn, data, "val")
        if best is None or v[SELECT] > best[0][SELECT]:
            best = (v, w, fn)
    return best
