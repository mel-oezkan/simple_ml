import torch.nn as nn
import torch
import numpy as np


class PositionalEmbeddings(nn.Module):
    def __init__(self, model_dim, base: int = 10_000):
        self.model_dim = model_dim
        self.dim_subset = np.arange(model_dim // 2).reshape((1, model_dim//2))
        self.base = base

    def forward(self, positions: torch.Tensor) -> torch.Tensor: 
        # check the shape of positions
        pos_shape = positions.shape
        assert pos_shape[-1] == 1

        pos_sin = np.sin(positions/(self.base**(2*self.dim_subset/self.model_dim)))
        pos_cos = np.cos(positions/(self.base**(2*self.dim_subset/self.model_dim)))

        embedding = np.empty((len(positions), self.model_dim))
        embedding[:, 0::2] = pos_sin
        embedding[:, 1::2] = pos_cos
        return embedding


class PatchEmbedding(nn.Module):
    def __init__(self):
        pass


class ViT(nn.Module):
    def __init__(self):
        pass