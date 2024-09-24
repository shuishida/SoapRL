"""Policies: abstract base class and concrete implementations."""

from functools import partial
from typing import Union, List, Dict, Type, Any, Tuple, Optional

from einops import repeat
from gymnasium import spaces

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from common.utils.model_utils import create_mlp
from dac.distributions import DistributionModule
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor, NatureCNN, CombinedExtractor
from stable_baselines3.common.type_aliases import Schedule, PyTorchObs


class PPOActorCriticPolicy(ActorCriticPolicy):
    """
    Policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        option_dim: int,
        lr_schedule: Schedule,
        net_arch: Union[List[int], Dict[str, List[int]], List[Dict[str, List[int]]], None] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.option_dim = option_dim

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                option_dim=self.option_dim,
            )
        )
        return data

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self.trans_features_extractor = self.make_features_extractor()

        self.mlp_trans = create_mlp(self.features_dim, net_arch=self.net_arch["pi"] if isinstance(self.net_arch, dict) else self.net_arch,
                                    activation=self.activation_fn, output_activation=self.activation_fn)
        self._build_mlp_extractor()

        self.action_dist = DistributionModule(self.action_space, self.mlp_extractor.latent_dim_pi, use_sde=self.use_sde, log_std_init=self.log_std_init,
                                              option_dim=self.option_dim, dist_kwargs=self.dist_kwargs)
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, self.option_dim)

        self.option_policy = nn.Sequential(
            nn.Linear(self.mlp_extractor.latent_dim_pi, self.option_dim),
            nn.LogSoftmax(dim=-1)
        )
        self.switch_prob = nn.Sequential(
            nn.Linear(self.mlp_extractor.latent_dim_pi, self.option_dim),
            nn.Sigmoid()
        )

        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.mlp_trans: np.sqrt(2),
                self.action_dist: 0.01,
                self.option_policy: 1.0,
                self.switch_prob: 1.0,
                self.value_net: 1,
            }
            if not self.share_features_extractor:
                # Note(antonin): this is to keep SB3 results
                # consistent, see GH#1148
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)
                module_gains[self.trans_features_extractor] = np.sqrt(2)

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)  # type: ignore[call-arg]

    def reset_noise(self, n_envs: int = 1) -> None:
        """
        Sample new weights for the exploration matrix.

        :param n_envs:
        """
        self.action_dist.reset_noise(n_envs)

    def forward(
        self,
        obs: PyTorchObs,
        prev_option: Optional[th.Tensor],
        episode_starts: th.Tensor,
        deterministic: bool = False
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        latent_pi, latent_vf, latent_obs = self.extract_latents(obs)

        if prev_option is None:
            prev_option = self.init_options(len(episode_starts), latent_obs)
        assert th.allclose(prev_option.sum(dim=-1), th.ones_like(prev_option.sum(dim=-1))), "Option probabilities must sum to 1"

        values = self.value_net(latent_vf)

        new_option_probs = self.compute_transitions(latent_obs, prev_option, episode_starts)
        with th.no_grad():
            new_option = F.gumbel_softmax((new_option_probs + 1e-8).log(), hard=True)

        distribution = self.action_dist(latent_pi, new_option)
        actions = distribution.get_actions(deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob, new_option, new_option_probs

    def init_options(self, batch_size: int, latent_obs) -> th.Tensor:
        log_option_prob = self.option_policy(latent_obs)  # log pi(z|s): shape (B, O)
        return F.gumbel_softmax(log_option_prob, hard=True)

    def extract_latents(self, obs: PyTorchObs, compute_pi: bool = True, compute_vf: bool = True, compute_trans: bool = True) \
        -> Tuple[Optional[th.Tensor], Optional[th.Tensor], Optional[th.Tensor]]:
        """
        Preprocess the observation if needed and extract features.

        :param obs: Observation
        :return: The extracted features. If features extractor is not shared, returns a tuple with the
            features for the actor and the features for the critic.
        """
        latent_pi, latent_vf, latent_trans = None, None, None
        if self.share_features_extractor:
            features = super(ActorCriticPolicy, self).extract_features(obs, self.features_extractor)
            if compute_pi and compute_vf:
                latent_pi, latent_vf = self.mlp_extractor(features)
            elif compute_pi:
                latent_pi = self.mlp_extractor.forward_actor(features)
            elif compute_vf:
                latent_vf = self.mlp_extractor.forward_critic(features)
            if compute_trans:
                latent_trans = self.mlp_trans(features)
        else:
            if compute_pi:
                pi_features = super(ActorCriticPolicy, self).extract_features(obs, self.pi_features_extractor)
                latent_pi = self.mlp_extractor.forward_actor(pi_features)
            if compute_vf:
                vf_features = super(ActorCriticPolicy, self).extract_features(obs, self.vf_features_extractor)
                latent_vf = self.mlp_extractor.forward_critic(vf_features)
            if compute_trans:
                trans_features = super(ActorCriticPolicy, self).extract_features(obs, self.trans_features_extractor)
                latent_trans = self.mlp_trans(trans_features)

        return latent_pi, latent_vf, latent_trans

    def compute_transitions(self, latent_obs: th.Tensor, prev_options: th.Tensor, episode_starts: th.Tensor):
        log_option_prob = self.option_policy(latent_obs)  # log pi(z|s): shape (B, O)
        switch_probs = self.switch_prob(latent_obs)       # p(switch|s): shape (B, O)
        switch_probs = th.where(repeat(episode_starts.bool(), "b -> b o", o=self.option_dim), th.ones_like(switch_probs), switch_probs)
        switch_prob = (switch_probs * prev_options).sum(dim=-1, keepdim=True)
        new_option_probs = (1 - switch_prob) * prev_options + switch_prob * log_option_prob.exp()
        return new_option_probs

    def evaluate_actions(self, obs: PyTorchObs, actions: th.Tensor,
                         option: th.Tensor, prev_option: th.Tensor, episode_starts: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        latent_pi, latent_vf, latent_obs = self.extract_latents(obs)
        values = self.value_net(latent_vf)
        distribution = self.action_dist(latent_pi, option)
        log_prob = distribution.log_prob(actions)
        new_option_probs = self.compute_transitions(latent_obs, prev_option, episode_starts)
        log_option_probs = (new_option_probs + 1e-8).log()
        log_option_prob = th.einsum("bo,bo->b", log_option_probs, option)
        entropy = distribution.entropy()
        option_entropy = -(new_option_probs * log_option_probs).sum(dim=-1)
        return values, log_prob, log_option_prob, entropy, option_entropy

    def get_distribution(self, obs: PyTorchObs) -> Distribution:
        raise NotImplementedError

    def predict_values(self, obs: PyTorchObs, prev_options: th.Tensor, episode_starts: th.Tensor) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        _, latent_vf, latent_obs = self.extract_latents(obs, compute_pi=False)

        values = self.value_net(latent_vf)
        new_option_probs = self.compute_transitions(latent_obs, prev_options, episode_starts)
        log_option_probs = (new_option_probs + 1e-8).log()
        with th.no_grad():
            new_option = F.gumbel_softmax(log_option_probs, hard=True)
        return values, new_option, new_option_probs

    def _predict(
        self,
        observation: PyTorchObs,
        prev_option: th.Tensor,
        episode_starts: th.Tensor,
        deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor]:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        actions, values, log_prob, log_option_prob, switch_prob, new_option, new_option_probs = self(observation, prev_option=prev_option, episode_starts=episode_starts, deterministic=deterministic)
        return actions, new_option

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        prev_option: Optional[np.ndarray] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param lstm_states: The last hidden and memory states for the LSTM.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the lstm states in that case).
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        observation, vectorized_env = self.obs_to_tensor(observation)

        with th.no_grad():
            if prev_option is not None:
                prev_option = th.tensor(prev_option, dtype=th.float32, device=self.device)

            assert episode_start is not None
            if not vectorized_env:
                episode_start = np.asarray([episode_start])
            episode_start = th.tensor(episode_start, dtype=th.float32, device=self.device)

            actions, option = self._predict(
                observation, prev_option=prev_option, episode_starts=episode_start, deterministic=deterministic
            )
            states = option.cpu().numpy()

        # Convert to numpy
        actions = actions.cpu().numpy()

        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.action_space.low, self.action_space.high)

        # Remove batch dimension if needed
        if not vectorized_env:
            actions = actions.squeeze(axis=0)

        return actions, states


class PPOActorCriticCnnPolicy(PPOActorCriticPolicy):
    def __init__(self, *args, features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN, **kwargs):
        super().__init__(*args, features_extractor_class=features_extractor_class, **kwargs)


class PPOMultiInputActorCriticPolicy(PPOActorCriticPolicy):
    def __init__(self, *args, features_extractor_class: Type[BaseFeaturesExtractor] = CombinedExtractor, **kwargs):
        super().__init__(*args, features_extractor_class=features_extractor_class, **kwargs)
