import os
import urllib.request
import zipfile

import numpy as np
import pandas as pd

URLS = {
    "ml-1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
    "lastfm-2k": "https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip",
    "personality-2018": "https://files.grouplens.org/datasets/personality-isf2018/personality-isf2018.zip",
}
EXTRACT_TO = {"ml-1m": ".", "lastfm-2k": "lastfm-2k", "personality-2018": "."}
FOLDERS = {"ml-1m": "ml-1m", "lastfm-2k": "lastfm-2k", "personality-2018": "personality-isf2018"}


def download(root, names=tuple(URLS)):
    os.makedirs(root, exist_ok=True)
    for name in names:
        if os.path.isdir(os.path.join(root, FOLDERS[name])):
            continue
        zp = os.path.join(root, URLS[name].split("/")[-1])
        if not os.path.exists(zp):
            print("downloading", URLS[name], flush=True)
            urllib.request.urlretrieve(URLS[name], zp)
        with zipfile.ZipFile(zp) as zf:
            zf.extractall(os.path.join(root, EXTRACT_TO[name]))


def _prepare(df, min_inter=5):
    df = df.drop_duplicates(["user", "item"])
    cnt = df.groupby("user").size()
    df = df[df["user"].isin(cnt[cnt >= min_inter].index)].copy()
    uids, iids = sorted(df["user"].unique()), sorted(df["item"].unique())
    df["u"] = df["user"].map({u: k for k, u in enumerate(uids)})
    df["i"] = df["item"].map({i: k for k, i in enumerate(iids)})
    return df, len(uids), len(iids)


def _split(df, n_users, n_items, seed):
    rng = np.random.default_rng(seed)
    train_u, train_i, val, test = [], [], {}, {}
    for u, grp in df.groupby("u"):
        items = grp["i"].tolist()
        rng.shuffle(items)
        n = len(items)
        nt = max(1, int(round(0.1 * n)))
        nv = max(1, int(round(0.1 * n)))
        test[u], val[u] = items[:nt], items[nt:nt + nv]
        tr = items[nt + nv:] or [val[u].pop()]
        train_u += [u] * len(tr)
        train_i += tr
    train_u = np.asarray(train_u, dtype=np.int64)
    train_i = np.asarray(train_i, dtype=np.int64)
    user_pos = [set() for _ in range(n_users)]
    for u, i in zip(train_u, train_i):
        user_pos[u].add(int(i))
    print(f"[data] users={n_users} items={n_items} train={len(train_u)} "
          f"val={sum(map(len, val.values()))} test={sum(map(len, test.values()))}", flush=True)
    return {"n_users": n_users, "n_items": n_items, "train_u": train_u, "train_i": train_i,
            "user_pos": user_pos, "val": val, "test": test}


def _read_ml1m(root):
    r = pd.read_csv(os.path.join(root, "ml-1m", "ratings.dat"), sep="::", engine="python",
                    names=["user", "item", "rating", "ts"], encoding="latin-1")
    r = r[r["rating"] >= 3.5]
    r["user"] = r["user"].astype(str)
    return r


def _read_personality(root):
    r = pd.read_csv(os.path.join(root, "personality-isf2018", "ratings.csv"))
    r.columns = [c.strip() for c in r.columns]
    r = r.rename(columns={"useri": "user", "movie_id": "item"})
    r = r[r["rating"] >= 3.5]
    r["user"] = r["user"].astype(str).str.strip()
    return r


def _read_lastfm(root):
    r = pd.read_csv(os.path.join(root, "lastfm-2k", "user_artists.dat"), sep="\t")
    r.columns = [c.strip() for c in r.columns]
    r = r.rename(columns={"userID": "user", "artistID": "item"})
    r["user"] = r["user"].astype(str)
    return r


READERS = {"ml-1m": _read_ml1m, "personality-2018": _read_personality, "lastfm-2k": _read_lastfm}


def load(name, seed, root="data"):
    download(root, (name,))
    df, nu, ni = _prepare(READERS[name](root))
    return _split(df, nu, ni, seed)
