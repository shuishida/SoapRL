"""Probability distributions."""

from copy import deepcopy
from typing import Any, Dict, Optional

import torch as th
from einops import rearrange
from gymnasium import spaces
from torch import nn
from torch.distributions import Bernoulli, Categorical, Normal

from stable_baselines3.common.distributions import Distribution, DiagGaussianDistribution, SquashedDiagGaussianDistribution, \
    CategoricalDistribution, MultiCategoricalDistribution, BernoulliDistribution, StateDependentNoiseDistribution
from stable_baselines3.common.preprocessing import get_action_dim


class DiagGaussianModule(nn.Module):
    def __init__(self, output_space: spaces.Box, latent_dim: int, log_std_init: float = 0.0, option_dim: Optional[int] = None, **kwargs):
        super().__init__()
        self.option_dim = option_dim
        action_dim = get_action_dim(output_space)
        self.dist = DiagGaussianDistribution(action_dim * option_dim)
        self.net, self.log_std = self.dist.proba_distribution_net(
            latent_dim=latent_dim, log_std_init=log_std_init
        )

    def forward(self, latent_pi, option) -> DiagGaussianDistribution:
        mean_actions = self.net(latent_pi)
        action_std = th.ones_like(mean_actions) * self.log_std.exp()
        dist = deepcopy(self.dist)
        if self.option_dim is not None:
            mean_actions = rearrange(mean_actions, "b (o a) -> b o a", o=self.option_dim)
            mean_actions = (mean_actions * option.unsqueeze(-1)).sum(dim=-2)
            action_std = rearrange(action_std, "b (o a) -> b o a", o=self.option_dim)
            action_std = (action_std * option.unsqueeze(-1)).sum(dim=-2)
        dist.distribution = Normal(mean_actions, action_std)
        return dist


class SquashedDiagGaussianModule(nn.Module):
    def __init__(self, output_space: spaces.Box, latent_dim: int, log_std_init: float = 0.0, option_dim: Optional[int] = None, **kwargs):
        super().__init__()
        self.option_dim = option_dim
        action_dim = get_action_dim(output_space)
        self.dist = SquashedDiagGaussianDistribution(action_dim if option_dim is None else action_dim * option_dim)
        self.net, self.log_std = self.dist.proba_distribution_net(
            latent_dim=latent_dim, log_std_init=log_std_init
        )

    def forward(self, latent_pi, option) -> SquashedDiagGaussianDistribution:
        mean_actions = self.net(latent_pi)
        action_std = th.ones_like(mean_actions) * self.log_std.exp()
        dist = deepcopy(self.dist)
        if self.option_dim is not None:
            mean_actions = rearrange(mean_actions, "b (o a) -> b o a", o=self.option_dim)
            mean_actions = (mean_actions * option.unsqueeze(-1)).sum(dim=-2)
            action_std = rearrange(action_std, "b (o a) -> b o a", o=self.option_dim)
            action_std = (action_std * option.unsqueeze(-1)).sum(dim=-2)
        dist.distribution = Normal(mean_actions, action_std)
        return dist


class CategoricalModule(nn.Module):
    def __init__(self, output_space: spaces.Discrete, latent_dim: int, option_dim: Optional[int] = None, **kwargs):
        super().__init__()
        self.option_dim = option_dim
        action_dim = output_space.n
        self.dist = CategoricalDistribution(action_dim if option_dim is None else action_dim * option_dim)
        self.net = self.dist.proba_distribution_net(latent_dim=latent_dim)

    def forward(self, latent_pi, option) -> CategoricalDistribution:
        action_logits = self.net(latent_pi)
        dist = deepcopy(self.dist)
        if self.option_dim is not None:
            action_logits = rearrange(action_logits, "b (o a) -> b o a", o=self.option_dim)
            action_logits = (action_logits * option.unsqueeze(-1)).sum(dim=-2)
        dist.distribution = Categorical(logits=action_logits)
        return dist


class MultiCategoricalModule(nn.Module):
    def __init__(self, output_space: spaces.MultiDiscrete, latent_dim: int, option_dim: Optional[int] = None, **kwargs):
        super().__init__()
        self.option_dim = option_dim
        self.action_dims = list(output_space.nvec) if option_dim is None else [dim * option_dim for dim in output_space.nvec]
        self.dist = MultiCategoricalDistribution(self.action_dims)
        self.net = self.dist.proba_distribution_net(latent_dim=latent_dim)

    def forward(self, latent_pi, option) -> MultiCategoricalDistribution:
        action_logits = self.net(latent_pi)
        dist = deepcopy(self.dist)
        categoricals = []
        for split in th.split(action_logits, self.action_dims, dim=1):
            if self.option_dim is not None:
                split = rearrange(split, "b (o a) -> b o a", o=self.option_dim)
                split = (split * option.unsqueeze(-1)).sum(dim=-2)
            categoricals.append(Categorical(logits=split))
        dist.distribution = categoricals
        return dist


class BernoulliModule(nn.Module):
    def __init__(self, output_space: spaces.MultiBinary, latent_dim: int, option_dim: Optional[int] = None, **kwargs):
        super().__init__()
        self.option_dim = option_dim
        action_dim = output_space.n
        self.dist = BernoulliDistribution(action_dim if option_dim is None else action_dim * option_dim)
        self.net = self.dist.proba_distribution_net(latent_dim=latent_dim)

    def forward(self, latent_pi, option) -> BernoulliDistribution:
        action_logits = self.net(latent_pi)
        dist = deepcopy(self.dist)
        if self.option_dim is not None:
            action_logits = rearrange(action_logits, "b (o a) -> b o a", o=self.option_dim)
            action_logits = (action_logits * option.unsqueeze(-1)).sum(dim=-2)
        dist.distribution = Bernoulli(logits=action_logits)
        return dist


class ModularStateDependentNoiseDistribution(StateDependentNoiseDistribution):
    """
    Distribution class for using generalized State Dependent Exploration (gSDE).
    Paper: https://arxiv.org/abs/2005.05719

    It is used to create the noise exploration matrix and
    compute the log probability of an action with that noise.

    :param action_dim: Dimension of the action space.
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,)
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this ensures bounds are satisfied.
    :param learn_features: Whether to learn features for gSDE or not.
        This will enable gradients to be backpropagated through the features
        ``latent_sde`` in the code.
    :param epsilon: small value to avoid NaN due to numerical imprecision.
    """

    def __init__(
        self,
        dist_module: 'StateDependentNoiseModule',
        action_dim: int,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        learn_features: bool = False,
        epsilon: float = 1e-6,
    ):
        self.dist_module = dist_module
        super().__init__(action_dim, full_std, use_expln, squash_output, learn_features, epsilon)

    def sample_weights(self, log_std: th.Tensor, batch_size: int = 1) -> None:
        pass

    def get_noise(self, latent_sde: th.Tensor) -> th.Tensor:
        return self.dist_module.get_noise(latent_sde)


class StateDependentNoiseModule(nn.Module):
    def __init__(
        self,
        output_space: spaces.Box,
        latent_dim: int,
        log_std_init: float = 0.0,
        option_dim: Optional[int] = None,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        learn_features: bool = False,
        epsilon: float = 1e-6,
        **kwargs
    ):
        super().__init__()
        self.option_dim = option_dim
        self.action_dim = get_action_dim(output_space) if option_dim is None else get_action_dim(output_space) * option_dim
        self.dist = SquashedDiagGaussianDistribution(self.action_dim)
        self.dist = ModularStateDependentNoiseDistribution(self, self.action_dim, full_std=full_std, use_expln=use_expln,
                                                           squash_output=squash_output, learn_features=learn_features, epsilon=epsilon)
        self.net, self.log_std = self.dist.proba_distribution_net(
            latent_dim=latent_dim, latent_sde_dim=latent_dim, log_std_init=log_std_init
        )
        self.latent_sde_dim = self.dist.latent_sde_dim

        self.use_expln = use_expln
        self.full_std = full_std
        self.epsilon = epsilon
        self.learn_features = learn_features

        self.weights_dist = None
        self.exploration_mat = None
        self.exploration_matrices = None
        self.sample_weights()

    def forward(self, latent_pi, option) -> ModularStateDependentNoiseDistribution:
        mean_actions = self.net(latent_pi)
        latent_sde = latent_pi
        dist = deepcopy(self.dist)
        # Stop gradient if we don't want to influence the features
        dist._latent_sde = latent_sde if self.learn_features else latent_sde.detach()
        variance = th.mm(dist._latent_sde**2, self.get_std(self.log_std) ** 2)
        if self.option_dim is not None:
            mean_actions = rearrange(mean_actions, "b (o a) -> b o a", o=self.option_dim)
            mean_actions = (mean_actions * option.unsqueeze(-1)).sum(dim=-2)
            variance = rearrange(variance, "b (o a) -> b o a", o=self.option_dim)
            variance = (variance * option.unsqueeze(-1)).sum(dim=-2)
        dist.distribution = Normal(mean_actions, th.sqrt(variance + self.epsilon))
        return dist

    def get_std(self, log_std: th.Tensor) -> th.Tensor:
        """
        Get the standard deviation from the learned parameter
        (log of it by default). This ensures that the std is positive.

        :param log_std:
        :return:
        """
        if self.use_expln:
            # From gSDE paper, it allows to keep variance
            # above zero and prevent it from growing too fast
            below_threshold = th.exp(log_std) * (log_std <= 0)
            # Avoid NaN: zeros values that are below zero
            safe_log_std = log_std * (log_std > 0) + self.epsilon
            above_threshold = (th.log1p(safe_log_std) + 1.0) * (log_std > 0)
            std = below_threshold + above_threshold
        else:
            # Use normal exponential
            std = th.exp(log_std)

        if self.full_std:
            return std
        # Reduce the number of parameters:
        return th.ones(self.latent_sde_dim, self.action_dim).to(log_std.device) * std

    def sample_weights(self, batch_size: int = 1) -> None:
        """
        Sample weights for the noise exploration matrix,
        using a centered Gaussian distribution.

        :param log_std:
        :param batch_size:
        """
        std = self.get_std(self.log_std)
        self.weights_dist = Normal(th.zeros_like(std), std)
        # Reparametrization trick to pass gradients
        self.exploration_mat = self.weights_dist.rsample()
        # Pre-compute matrices in case of parallel exploration
        self.exploration_matrices = self.weights_dist.rsample((batch_size,))

    def get_noise(self, latent_sde: th.Tensor) -> th.Tensor:
        latent_sde = latent_sde if self.learn_features else latent_sde.detach()
        # Default case: only one exploration matrix
        if len(latent_sde) == 1 or len(latent_sde) != len(self.exploration_matrices):
            return th.mm(latent_sde, self.exploration_mat)
        # Use batch matrix multiplication for efficient computation
        # (batch_size, n_features) -> (batch_size, 1, n_features)
        latent_sde = latent_sde.unsqueeze(dim=1)
        # (batch_size, 1, n_actions)
        noise = th.bmm(latent_sde, self.exploration_matrices)
        return noise.squeeze(dim=1)


class DistributionModule(nn.Module):
    def __init__(
        self,
        output_space: spaces.Space,
        latent_dim: int,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        option_dim: Optional[int] = None,
        dist_kwargs: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        self.output_space = output_space
        self.latent_dim = latent_dim

        if dist_kwargs is None:
            dist_kwargs = {}

        if isinstance(output_space, spaces.Box):
            module = StateDependentNoiseModule if use_sde else DiagGaussianModule
        elif isinstance(output_space, spaces.Discrete):
            module = CategoricalModule
        elif isinstance(output_space, spaces.MultiDiscrete):
            module = MultiCategoricalModule
        elif isinstance(output_space, spaces.MultiBinary):
            module = BernoulliModule
        else:
            raise NotImplementedError(
                "Error: probability distribution, not implemented for action space"
                f"of type {type(output_space)}."
                " Must be of type Gym Spaces: Box, Discrete, MultiDiscrete or MultiBinary."
            )
        self.dist_module = module(output_space, latent_dim, log_std_init=log_std_init, option_dim=option_dim, **dist_kwargs)

    def forward(self, latent_pi, option) -> Distribution:
        return self.dist_module(latent_pi, option)

    def reset_noise(self, n_envs: int = 1) -> None:
        """
        Sample new weights for the exploration matrix.

        :param n_envs:
        """
        assert isinstance(self.dist_module, StateDependentNoiseModule), "reset_noise() is only available when using gSDE"
        self.dist_module.sample_weights(n_envs)


def assert_is_onehot(features, dim=-1):
    ones = th.ones_like(features).mean(dim=dim)
    assert th.equal(features.sum(dim=dim), ones)
    assert th.equal(features.max(dim=dim)[0], ones)
    assert th.equal(features.min(dim=dim)[0], ones * 0)
