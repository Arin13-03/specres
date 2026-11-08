"""Builds the LaTeX result tables and a plain-text summary from results/<dataset>/*.json."""
import argparse
import glob
import os
import re

import numpy as np
from scipy.stats import ttest_rel

from common import load_json, SELECT
from specres.config import DATASETS, SEEDS

NAMES = {"lastfm-2k": "LastFM", "ml-1m": "ML-1M", "personality-2018": "Personality"}
ROWS = [("mf", "MF-BPR"), ("lightgcn", "LightGCN"), ("sgl", "SGL"), ("ncl", "NCL"), ("simgcl", "SimGCL"),
        ("lightgcl", "LightGCL"), ("ntssm", "NT-SSM"), ("xsimgcl", "XSimGCL"), ("gfcf", "GF-CF"),
        ("turbocf", "Turbo-CF"), ("fusion", "XSimGCL+GF-CF (dense)"), ("specres_heldout", "SpecRes (held-out fit)"),
        ("specres_xsimgcl", "SpecRes (ours)")]
GROUP_START = {"lightgcn", "gfcf", "specres_heldout"}


def runs(res, ds, key):
    """Per-seed results of a method; for grids, the configuration with the best seed-42 validation score."""
    d = os.path.join(res, ds)
    if os.path.exists(os.path.join(d, f"{key}__s42.json")):
        return {s: load_json(os.path.join(d, f"{key}__s{s}.json")) for s in SEEDS
                if os.path.exists(os.path.join(d, f"{key}__s{s}.json"))}
    cands = sorted(glob.glob(os.path.join(d, f"{key}_c*__s42.json")))
    if not cands:
        return {}
    best = max(cands, key=lambda p: load_json(p)["val_" + SELECT])
    stem = re.sub(r"__s42\.json$", "", os.path.basename(best))
    return {s: load_json(os.path.join(d, f"{stem}__s{s}.json")) for s in SEEDS
            if os.path.exists(os.path.join(d, f"{stem}__s{s}.json"))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="tables")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    datasets = [ds for ds in DATASETS if os.path.isdir(os.path.join(args.results, ds))]
    R = {ds: {k: runs(args.results, ds, k) for k, _ in ROWS} for ds in datasets}
    metric = {"R": ("test_Recall@20", "pu_recall20"), "N": ("test_NDCG@20", "pu_ndcg20")}
    mean = lambda ds, k, m: np.mean([r[metric[m][0]] for r in R[ds][k].values()]) if R[ds][k] else np.nan
    baselines = [k for k, _ in ROWS if not k.startswith("specres")]

    def significant(ds, k, m):
        ref = max((b for b in baselines if R[ds][b]), key=lambda b: mean(ds, b, m))
        tests = [ttest_rel(R[ds][k][s][metric[m][1]], R[ds][ref][s][metric[m][1]]) for s in R[ds][k] if s in R[ds][ref]]
        better = mean(ds, k, m) > mean(ds, ref, m)
        return ref, better and len(tests) == len(SEEDS) and all(t.pvalue < 0.01 for t in tests)

    lines = [r"\begin{table*}[!t]", r"\centering\small",
             r"\caption{Recall@20 (R) and NDCG@20 (N), mean of 3 seeds. \textbf{Bold}: best; \underline{underline}: "
             r"second. Gain: SpecRes over the best baseline. $^\dagger$: $p<0.01$ vs.\ the best baseline on every seed.}",
             r"\label{tab:main}", r"\begin{tabular}{l" + "cc" * len(datasets) + "}", r"\toprule",
             "Method & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{NAMES[d]}}}" for d in datasets) + r" \\",
             " & " + " & ".join(["R@20 & N@20"] * len(datasets)) + r" \\", r"\midrule"]
    for k, label in ROWS:
        if k in GROUP_START:
            lines.append(r"\midrule")
        cells = []
        for ds in datasets:
            for m in ("R", "N"):
                vals = sorted({round(mean(ds, kk, m), 4) for kk, _ in ROWS if R[ds][kk]}, reverse=True)
                mu = mean(ds, k, m)
                if np.isnan(mu):
                    cells.append("--")
                    continue
                c = f"{mu:.4f}"
                c = rf"\textbf{{{c}}}" if round(mu, 4) == vals[0] else (rf"\underline{{{c}}}" if round(mu, 4) == vals[1] else c)
                if k.startswith("specres") and significant(ds, k, m)[1]:
                    c += r"$^\dagger$"
                cells.append(c)
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    gains = []
    for ds in datasets:
        for m in ("R", "N"):
            ref = significant(ds, "specres_xsimgcl", m)[0] if R[ds]["specres_xsimgcl"] else None
            gains.append(f"{100 * (mean(ds, 'specres_xsimgcl', m) / mean(ds, ref, m) - 1):+.1f}\\%" if ref else "--")
    lines += [r"\midrule", r"Gain vs.\ best baseline & " + " & ".join(gains) + r" \\", r"\bottomrule",
              r"\end{tabular}", r"\end{table*}"]
    open(os.path.join(args.out, "table_main.tex"), "w").write("\n".join(lines) + "\n")

    summary = ["Per-method mean and std over seeds (test).", ""]
    for ds in datasets:
        summary.append(f"== {NAMES[ds]}")
        for k, label in ROWS:
            if R[ds][k]:
                r = [x["test_Recall@20"] for x in R[ds][k].values()]
                n = [x["test_NDCG@20"] for x in R[ds][k].values()]
                summary.append(f"{label:24s} R@20 {np.mean(r):.4f}±{np.std(r):.4f}  N@20 {np.mean(n):.4f}±{np.std(n):.4f}"
                               f"  seeds={len(r)}")
        a = os.path.join(args.results, ds, "analysis.json")
        if os.path.exists(a):
            summary.append("analysis: " + str({k: v for k, v in load_json(a).items() if k != "filter"}))
        summary.append("")
    open(os.path.join(args.out, "summary.txt"), "w").write("\n".join(summary) + "\n")
    print("wrote", os.path.join(args.out, "table_main.tex"), "and", os.path.join(args.out, "summary.txt"))


if __name__ == "__main__":
    main()
