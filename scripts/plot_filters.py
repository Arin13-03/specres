"""Plots the learned SpecRes responses saved by run_analysis.py."""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import load_json  # noqa: E402

SERIES = [("ml-1m", "ML-1M", "#2a78d6", "-"), ("personality-2018", "Personality", "#eb6834", "--"),
          ("lastfm-2k", "LastFM", "#1baf7a", ":")]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="tables/fig_filters.pdf")
    args = ap.parse_args()
    plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.linewidth": 0.5, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.33, 1.55))
    for ds, label, color, ls in SERIES:
        path = os.path.join(args.results, ds, "analysis.json")
        if not os.path.exists(path):
            continue
        f = load_json(path)["filter"]
        h = np.array(f["response"])
        ax.plot(f["sigma"], h / np.abs(h).max(), color=color, ls=ls, lw=1.2, label=label)
    ax.axhline(0, color="#9a9a94", lw=0.4)
    ax.set_xlabel(r"normalised singular value $\sigma/\sigma_{max}$ ($\leftarrow$ low frequency)")
    ax.set_ylabel("learned weight (scaled)")
    ax.invert_xaxis()
    ax.grid(axis="y", color="#e2e1dc", lw=0.4)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=6)
    fig.tight_layout(pad=0.3)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
