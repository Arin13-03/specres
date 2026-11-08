import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from specres.config import DATASETS, SEEDS, SELECT, TRAIN, load_json, save_json  # noqa: E402

XSIMGCL_GRID = [(0.02, 0.1), (0.05, 0.1), (0.1, 0.1), (0.2, 0.1), (0.05, 0.2), (0.1, 0.2)]

BASELINES = {
    "mf": [dict(model="mf", layers=0)],
    "sgl": [dict(model="sgl", edge_drop=0.1, cl_temp=0.2, cl_weight=0.1)],
    "ncl": [dict(model="ncl", hyper_layers=1, cl_temp=0.1, n_clusters=1000, cl_weight=w, proto_weight=0.8 * w)
            for w in (0.005, 0.05)],
    "simgcl": [dict(model="simgcl", cl_temp=0.2, cl_weight=w, eps=e) for w, e in ((0.1, 0.1), (0.5, 0.1))],
    "lightgcl": [dict(model="lightgcl", layers=2, svd_q=5, cl_temp=0.2, edge_drop=0.0, cl_weight=w) for w in (0.02, 0.2)],
    "ntssm": [dict(model="ntssm", layers=2, batch_size=2048, init="xavier", temp=0.2,
                   a_uu=1.2, a_ii=0.8, a_ui=0.8, a_iu=1.0)],
}
SIMGCL_EXTENDED = [dict(model="simgcl", cl_temp=0.2, cl_weight=w, eps=e) for w, e in ((0.02, 0.1), (0.02, 0.2), (0.1, 0.2))]


def xsimgcl_grid():
    return [dict(model="xsimgcl", cl_temp=0.2, cl_layer=1, cl_weight=w, eps=e) for w, e in XSIMGCL_GRID]


def parser(desc):
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--dataset", choices=DATASETS, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default="results")
    p.add_argument("--quick", action="store_true", help="2-epoch smoke test")
    return p


def train_cfg(args, **cfg):
    base = dict(TRAIN)
    if args.quick:
        base.update(epochs=2, min_epochs=1, eval_every=1, patience=1)
    return {**base, **cfg}


def paths(args, name, seed):
    d = os.path.join(args.out, args.dataset)
    return os.path.join(d, f"{name}__s{seed}.json"), os.path.join(d, "models", f"{name}__s{seed}.pt")


def done(args, name, seed):
    return os.path.exists(paths(args, name, seed)[0])


def selection_path(args):
    return os.path.join(args.out, args.dataset, "selection.json")


def get_selection(args):
    p = selection_path(args)
    return load_json(p) if os.path.exists(p) else {}


def set_selection(args, key, value):
    sel = get_selection(args)
    sel[key] = value
    save_json(selection_path(args), sel)


__all__ = ["BASELINES", "SIMGCL_EXTENDED", "xsimgcl_grid", "parser", "train_cfg", "paths", "done", "get_selection",
           "set_selection", "load_json", "save_json", "SELECT"]
