import random
from nanoid import generate
import numpy as np
import torch

def set_seed(seed: int | None = None):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

def get_generationid():
    generation_id = generate()
    return "runSampler-" + generation_id 