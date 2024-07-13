from functools import partial
from typing import Union, List, Dict, Type, Any, Tuple, Optional

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from gymnasium import spaces

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from common.utils.model_utils import create_mlp
from ppoem.distributions import EmbedActions, create_mixed_distribution, DistributionModule, ModularStateDependentNoiseDistribution
from ppoem.models import FeatureMixer
from stable_baselines3.common.distributions import Distribution, DiagGaussianDistribution, StateDependentNoiseDistribution, \
    SquashedDiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor, NatureCNN, CombinedExtractor
from stable_baselines3.common.type_aliases import Schedule, PyTorchObs


class TransitionModel(nn.Module):
    def __init__(self, action_space, obs_dim: int, latent_dim: int, option_channels: int, option_dim: int, n_layers: int,
                 activation_fn: Type[nn.Module] = nn.ReLU, switch_temp = -1.0):
        super().__init__()
        self.option_channels = option_channels
        self.option_dim = option_dim
        self.embed_actions = EmbedActions(action_space)
        self.mix_net = FeatureMixer([obs_dim, self.embed_actions.output_dim], latent_dim, n_layers, activation_fn)
        self.transition_matrix = nn.Sequential(
            nn.Linear(self.mix_net.combined_dim, option_channels * option_dim * option_dim),
            Rearrange("b (c oc on) -> b c oc on", oc=self.option_dim, on=self.option_dim)
        )
        self.switch_prob = nn.Linear(self.mix_net.combined_dim, option_channels)
        self.switch_temp = switch_temp

    def forward(self, latent_obs, actions, eps=1e-8):
        actions = self.embed_actions(actions)
        latent = self.mix_net(latent_obs, actions)
        matrix = self.transition_matrix(latent)
        switch_prob = th.sigmoid(self.switch_prob(latent) + self.switch_temp).unsqueeze(-1).unsqueeze(-1)
        identity = th.eye(self.option_dim, device=latent.device).unsqueeze(0).unsqueeze(0)
        transition = identity * (1 - switch_prob) + switch_prob * th.softmax(matrix, dim=-1)   # normalise probability p(z' | z; s, a)
        # print(transition)
        # return identity
        return (1 - eps) * transition + eps / self.option_dim


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
        option_channels: int,
        option_dim: int,
        mixed_features_dim: int,
        n_mix_layers: int,
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
        self.option_channels = option_channels
        self.option_dim = option_dim
        self.mixed_features_dim = mixed_features_dim
        self.n_mix_layers = n_mix_layers

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
                option_channels=self.option_channels,
                option_dim=self.option_dim,
                mixed_features_dim=self.mixed_features_dim,
                n_mix_layers=self.n_mix_layers,
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
                                              option_dim=self.option_dim * self.option_channels, dist_kwargs=self.dist_kwargs)
        self.transition_net = TransitionModel(self.action_space, self.mlp_extractor.latent_dim_pi, self.mixed_features_dim,
                                              self.option_channels, self.option_dim, self.n_mix_layers, self.activation_fn)
        self.value_net = nn.Sequential(
            nn.Linear(self.mlp_extractor.latent_dim_vf, self.option_channels * self.option_dim),
            Rearrange("b (c o) -> b c o", c=self.option_channels)
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
                self.transition_net: 1.0,
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
        option_prob: th.Tensor,
        deterministic: bool = False
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        assert th.allclose(option_prob.sum(dim=-1), th.ones_like(option_prob.sum(dim=-1))), "Option probabilities must sum to 1"

        latent_pi, latent_vf, latent_trans = self.extract_latents(obs)
        values = self.value_net(latent_vf)
        distribution = self.action_dist(latent_pi)
        actions = self.sample_actions(distribution, option_prob, deterministic=deterministic)

        log_prob, policy_matrix, curr_next_option = self.compute_transitions(distribution, actions, latent_trans)
        new_option = th.einsum("bci,bcij->bcj", option_prob, policy_matrix)
        action_forward_prob = new_option.sum(dim=-1, keepdim=True)             # normalisation constant alpha_t
        new_option_prob = new_option / action_forward_prob                     # (B, C, O) zeta(z')

        return actions, values, log_prob, new_option_prob, policy_matrix, curr_next_option, action_forward_prob.squeeze(-1)

    def init_options(self, batch_size: int, curr_obs) -> th.Tensor:
        # init_options = th.zeros((batch_size, self.option_channels, self.option_dim), device=self.device)
        # init_options[curr_obs == 1, :, 0] = 1
        # init_options[curr_obs == 2, :, 1] = 1
        # return init_options
        return th.ones((batch_size, self.option_channels, self.option_dim), device=self.device) / self.option_dim

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

    def compute_transitions(self, distribution: Distribution, actions: th.Tensor, latent_trans: th.Tensor):
        log_prob = distribution.log_prob(repeat(actions, "b ... -> (b c o) ...", c=self.option_channels, o=self.option_dim))
        log_prob = rearrange(log_prob, "(b c o) -> b c o", c=self.option_channels, o=self.option_dim)
        curr_next_option = self.transition_net(latent_trans, actions)       # p(z'| s, z): shape (B, C, O, O)
        policy_matrix = th.einsum("bci,bcij->bcij", log_prob.exp(), curr_next_option)    # p(a, z'| s, z): shape (B, C, O, O)
        return log_prob, policy_matrix, curr_next_option

    def evaluate_actions(self, obs: PyTorchObs, actions: th.Tensor, option_prob: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        latent_pi, latent_vf, latent_trans = self.extract_latents(obs)
        values = self.value_net(latent_vf)
        distribution = self.action_dist(latent_pi)
        log_prob, policy_matrix, curr_next_option = self.compute_transitions(distribution, actions, latent_trans)
        entropy = rearrange(distribution.entropy(), "(b c o) -> b c o", c=self.option_channels, o=self.option_dim)
        return values, log_prob, curr_next_option, entropy

    def get_distribution(self, obs: PyTorchObs) -> Distribution:
        raise NotImplementedError

    def sample_actions(self,
        distribution: Distribution,
        option_prob: th.Tensor,
        deterministic: bool = False) -> th.Tensor:
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        B, C, O = option_prob.shape
        assert C == self.option_channels and O == self.option_dim, f"Option probabilities must have shape (B, C, O) == ({len(option_prob) // self.option_dim}, {self.option_channels}, {self.option_dim}), got shape {option_prob.shape}"
        option_prob = option_prob.mean(dim=1)

        if isinstance(distribution, (DiagGaussianDistribution, SquashedDiagGaussianDistribution,
                                     StateDependentNoiseDistribution, ModularStateDependentNoiseDistribution)):
            actions = distribution.get_actions(deterministic)   # (b o) ...)
            actions = rearrange(actions, "(b o) ... -> b o ...", o=self.option_dim)
            if not deterministic:
                option_prob = F.gumbel_softmax(option_prob.log(), hard=True)
            option_index = th.argmax(option_prob, dim=-1)
            sampled_actions = actions[th.arange(B, device=actions.device), option_index]
            return sampled_actions
        else:
            mixed_distribution = create_mixed_distribution(distribution, option_prob)
            actions = mixed_distribution.get_actions(deterministic)
            return actions
        # batch_size = option_prob.shape[0]
        # batch_index = th.arange(batch_size, dtype=th.long, device=option_prob.device)
        # channel_index = th.randint(self.option_channels, (batch_size,), device=option_prob.device, dtype=th.long)
        # option_prob = option_prob[batch_index, channel_index]
        # option_prob = option_prob if deterministic else F.gumbel_softmax(option_prob.log(), hard=True)
        # option_index = th.argmax(option_prob, dim=-1)
        # actions = rearrange(distribution.get_actions(deterministic), "(b c o) ... -> b c o ...", c=self.option_channels, o=self.option_dim)
        # return actions[batch_index, channel_index, option_index]     # (B)

    def predict_values(self, obs: PyTorchObs) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        _, latent_vf, _ = self.extract_latents(obs, compute_pi=False)
        return self.value_net(latent_vf)

    def _predict(
        self,
        observation: PyTorchObs,
        option_prob: th.Tensor,
        deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor]:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        actions, values, log_prob, new_option_prob, policy_matrix, curr_next_options, norm = self(observation, option_prob=option_prob, deterministic=deterministic)
        return actions, new_option_prob

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[np.ndarray] = None,
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

        if isinstance(observation, dict):
            n_envs = observation[list(observation.keys())[0]].shape[0]
        else:
            n_envs = observation.shape[0]

        with th.no_grad():
            init_option_prob = self.init_options(n_envs, observation)

            if state is None:
                option_prob = init_option_prob.clone()
            else:
                option_prob = th.tensor(state, dtype=th.float32, device=self.device)

            if episode_start is not None:
                if not vectorized_env:
                    episode_start = np.asarray([episode_start])
                episode_start = th.tensor(episode_start, dtype=th.float32, device=self.device)
                option_prob = th.where(repeat(episode_start, "b -> b c o", c=self.option_channels, o=self.option_dim).bool(), init_option_prob, option_prob)

            actions, option_prob = self._predict(
                observation, option_prob=option_prob, deterministic=deterministic
            )
            states = option_prob.cpu().numpy()

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
