from typing import Type, List
import torch.nn as nn


def create_mlp(
    input_dim: int,
    output_dim: int = None,
    net_arch: List[int] = (),
    activation: Type[nn.Module] = nn.ReLU,
    with_bias: bool = True,
    output_activation=nn.Identity,
    dropout=0.
):
    if output_dim is not None:
        net_arch = (*net_arch, output_dim)
    sizes = [input_dim, *net_arch]
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1], bias=with_bias), nn.Dropout(dropout), act()]
    return nn.Sequential(*layers)
