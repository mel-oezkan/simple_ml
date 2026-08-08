import random
import numpy as np
import torch

def set_seed(seed: int | None = None):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
