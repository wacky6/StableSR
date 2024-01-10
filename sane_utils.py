import torch
import numpy as np
import random
import os

def seed_everything(seed: int) -> int:
    """Seed RNGs without polluting STDOUT
    """

    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.environ["PL_SEED_WORKERS"] = f"{1}"

    return seed