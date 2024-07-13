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
    old_policy_matrix: th.Tensor
    old_curr_next_options: th.Tensor
    returns: th.Tensor
    action_forward: th.Tensor
    option_forward: th.Tensor
    option_backward: th.Tensor
    option_advantages: th.Tensor
    option_joint_probs: th.Tensor
    advantage_matrix: th.Tensor
