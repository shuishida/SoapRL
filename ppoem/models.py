from typing import Type, List

import torch as th
import torch.nn as nn


class FeatureMixer(nn.Module):
    def __init__(self,
                 feature_dims: List[int], latent_dim: int = None, n_layers: int = None,
                 activation_fn: Type[nn.Module] = nn.ReLU):
        super().__init__()
        self.feature_dims = feature_dims
        self.latent_dim = latent_dim
        self.combined_dim = sum(feature_dims) + latent_dim

        self.init_layer = nn.Sequential(nn.Linear(sum(feature_dims), latent_dim), activation_fn())
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(self.combined_dim, latent_dim), activation_fn())
            for _ in range(n_layers - 1)
        ])

    def forward(self, *args: th.Tensor) -> th.Tensor:
        latents = self.init_layer(th.cat(args, dim=-1))
        for layer in self.layers:
            latents = layer(th.cat([*args, latents], dim=-1))
        return th.cat([*args, latents], dim=-1)
