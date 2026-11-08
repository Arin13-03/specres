import json
import os
import random

import numpy as np
import torch

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

K_LIST = (10, 20, 50)
SELECT = "Recall@20"
DATASETS = ("lastfm-2k", "ml-1m", "personality-2018")
SEEDS = (42, 123, 456)

TRAIN = dict(dim=64, layers=3, lr=1e-3, batch_size=4096, l2=1e-4,
             epochs=400, min_epochs=30, eval_every=5, patience=4)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def free_memory():
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE.type == "mps":
        torch.mps.empty_cache()


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, default=float)
    os.replace(tmp, path)


def load_json(path):
    with open(path) as f:
        return json.load(f)
