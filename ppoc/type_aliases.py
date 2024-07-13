"""Common aliases for type hints"""

from typing import NamedTuple

import torch as th
from stable_baselines3.common.type_aliases import PyTorchObs


class PPORolloutBufferSamples(NamedTuple):
    observations: PyTorchObs
    next_observations: PyTorchObs
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    option: th.Tensor