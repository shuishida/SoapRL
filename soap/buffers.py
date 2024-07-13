from typing import Union, Optional, Generator

from einops import repeat
from gymnasium import spaces
import torch as th
import numpy as np

from soap.type_aliases import PPORolloutBufferSamples
from stable_baselines3.common.buffers import BaseBuffer
from stable_baselines3.common.type_aliases import GymObs
from stable_baselines3.common.vec_env import VecNormalize


class PPORolloutBuffer(BaseBuffer):
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.
    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        option_channels: int,
        option_dim: int,
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):

        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        self.option_channels = option_channels
        self.option_dim = option_dim
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.reset()
        self.count = 0

    def reset(self) -> None:
        if isinstance(self.obs_shape, dict):
            self.observations = {}
            self.next_observations = {}
            for key, obs_input_shape in self.obs_shape.items():
                self.observations[key] = th.zeros((self.buffer_size, self.n_envs, *obs_input_shape), dtype=th.float)
                self.next_observations[key] = th.zeros((self.buffer_size, self.n_envs, *obs_input_shape), dtype=th.float)
        else:
            self.observations = th.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=th.float)
            self.next_observations = th.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=th.float)

        self.actions = th.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=th.float)
        self.rewards = th.zeros((self.buffer_size, self.n_envs), dtype=th.float)
        self.dones = th.zeros((self.buffer_size, self.n_envs), dtype=th.float)
        self.terminals = th.zeros((self.buffer_size, self.n_envs), dtype=th.float)
        self.values = th.zeros((self.buffer_size, self.n_envs, self.option_channels, self.option_dim), dtype=th.float)
        self.next_values = th.zeros((self.buffer_size, self.n_envs, self.option_channels, self.option_dim), dtype=th.float)
        self.returns = th.zeros((self.buffer_size, self.n_envs, self.option_channels, self.option_dim), dtype=th.float)
        self.advantages = th.zeros((self.buffer_size, self.n_envs, self.option_channels, self.option_dim), dtype=th.float)
        self.log_probs = th.zeros((self.buffer_size, self.n_envs, self.option_channels, self.option_dim), dtype=th.float)

        self.advantage_matrix = th.zeros((self.buffer_size, self.n_envs, self.option_channels, self.option_dim, self.option_dim), dtype=th.float)

        self.action_forward = th.zeros((self.buffer_size, self.n_envs, self.option_channels), dtype=th.float)                        # alpha_t
        self.option_forward = th.zeros((self.buffer_size, self.n_envs, self.option_channels, self.option_dim), dtype=th.float)          # zeta(z_t)
        self.option_backward = th.zeros((self.buffer_size, self.n_envs, self.option_channels, self.option_dim), dtype=th.float)          # beta(z_{t+1})
        self.next_option_forward = th.zeros((self.buffer_size, self.n_envs, self.option_channels, self.option_dim), dtype=th.float)     # zeta(z_{t+1})

        self.policy_matrix = th.zeros((self.buffer_size, self.n_envs, self.option_channels, self.option_dim, self.option_dim), dtype=th.float)        # p(a, z'| s, z)
        self.curr_next_options = th.zeros((self.buffer_size, self.n_envs, self.option_channels, self.option_dim, self.option_dim), dtype=th.float)

        self.option_advantages = th.ones((self.buffer_size, self.n_envs, self.option_channels, self.option_dim), dtype=th.float)    # beta(z_{t+1})
        self.option_joint_probs = th.zeros((self.buffer_size, self.n_envs, self.option_channels, self.option_dim, self.option_dim), dtype=th.float)   # pi(z_t, z_{t+1} | \tau)

        super().reset()

    def compute_returns_and_advantage(self, last_values: th.Tensor, save_dir) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
        where R is the sum of discounted reward with value bootstrap
        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        """
        # print("Save directory: ", save_dir)
        # os.makedirs(save_dir, exist_ok=True)

        self.next_values[-1] = last_values.cpu().clone().detach()

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            next_values = self.next_values[step] * (1.0 - self.terminals[step].view(self.n_envs, 1, 1))   # (N, C, Z')
            target_values = self.rewards[step].view(-1, 1, 1) + self.gamma * next_values + self.gamma * self.gae_lambda * (1.0 - self.dones[step].view(self.n_envs, 1, 1)) * last_gae_lam   # V_target (B, Z')
            self.advantage_matrix[step] = advantage_matrix = target_values.unsqueeze(-2) - self.values[step].unsqueeze(-1)   # A(s, z, z') = V_target(s, z') - V(s, z)
            self.advantages[step] = last_gae_lam = th.einsum("bcij,bcij->bci", self.curr_next_options[step], advantage_matrix)  # (B, Z)
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values

        backward = th.ones((self.n_envs, self.option_channels, self.option_dim), dtype=th.float)
        for step in reversed(range(self.buffer_size)):
            self.option_backward[step] = backward = th.where(repeat(self.dones[step].bool(), "b -> b c j", c=self.option_channels, j=self.option_dim), th.ones_like(backward), backward)
            # \sum_{z'} [\beta(z')\pi(a, z'|s, z)]
            backward = th.einsum("bcj,bcij,bc->bci", backward, self.policy_matrix[step], 1 / self.action_forward[step])

        self.option_joint_probs = th.einsum("tbci,tbcij,tbcj,tbc->tbcij", self.option_forward, self.policy_matrix, self.option_backward, 1 / self.action_forward)
        assert th.allclose(self.option_joint_probs.sum(dim=(-1, -2)), th.ones((self.buffer_size, self.n_envs, self.option_channels), dtype=th.float)), \
            "Option joint probability should sum to 1."

        option_utility_norm = th.zeros((self.n_envs, self.option_channels, self.option_dim), dtype=th.float)
        for step in reversed(range(self.buffer_size)):
            self.option_advantages[step] = option_advantage = (self.advantages[step] * self.option_forward[step]).sum(dim=-1, keepdim=True) + (1.0 - self.dones[step].view(self.n_envs, 1, 1)) * option_utility_norm
            option_utility = th.einsum("bcj,bcij,bc->bci", option_advantage, self.policy_matrix[step], 1 / self.action_forward[step])
            option_utility_norm = option_utility - (option_utility * self.option_forward[step]).sum(dim=-1, keepdim=True)

        # if self.count % 10 == 0:
        #     data = {"obs": self.observations, "actions": self.actions, "rewards": self.rewards, "dones": self.dones, "values": self.values, "log_probs": self.log_probs,
        #             "returns": self.returns, "advantages": self.advantages, "next_values": self.next_values, "action_forward": self.action_forward,
        #             "option_forward": self.option_forward, "option_advantages": self.option_advantages, "policy_matrix": self.policy_matrix, "curr_next_options": self.curr_next_options, "option_joint_probs": self.option_joint_probs}
        #
        #     index = list(self.dones[:, 0]).index(1.0) + 1
        #     np.save(Path(save_dir) / f"{self.count}_rollout.npy", {k: v[index:index + 22, 0].numpy() for k, v in data.items()})
        self.count += 1

    def add(
        self,
        obs: GymObs,
        next_obs: GymObs,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        terminal: th.Tensor,
        value: th.Tensor,
        log_prob: th.Tensor,
        option_forward: th.Tensor,
        next_option_forward: th.Tensor,
        policy_matrix: th.Tensor,
        curr_next_option: th.Tensor,
        action_forward: th.Tensor,
        next_value: Optional[th.Tensor] = None
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Dict):
            for key in self.observations.keys():
                obs_ = obs[key]
                if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                    obs_ = obs_.reshape((self.n_envs, *self.obs_shape[key]))
                self.observations[key][self.pos] = obs_

                next_obs_ = next_obs[key]
                if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                    next_obs_ = next_obs_.reshape((self.n_envs, *self.obs_shape[key]))
                self.next_observations[key][self.pos] = next_obs_
        else:
            if isinstance(self.observation_space, spaces.Discrete):
                obs = obs.reshape((self.n_envs,) + self.obs_shape)
                next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))
            self.observations[self.pos] = obs
            self.next_observations[self.pos] = next_obs

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.actions[self.pos] = th.tensor(action)
        self.rewards[self.pos] = th.tensor(reward)
        self.dones[self.pos] = th.tensor(done)
        self.terminals[self.pos] = terminal.cpu().clone().detach()
        self.values[self.pos] = value = value.cpu().clone().detach()
        self.log_probs[self.pos] = log_prob.cpu().clone().detach()

        self.action_forward[self.pos] = action_forward.cpu().clone().detach()
        self.option_forward[self.pos] = option_forward.cpu().clone().detach()
        self.next_option_forward[self.pos] = next_option_forward.cpu().clone().detach()
        self.policy_matrix[self.pos] = policy_matrix.cpu().clone().detach()
        self.curr_next_options[self.pos] = curr_next_option.cpu().clone().detach()

        if next_value is not None:
            self.next_values[self.pos] = next_value.cpu().clone().detach()

        if self.pos > 0:
            self.next_values[self.pos-1] = th.where(repeat(self.dones[self.pos-1].bool(), "b -> b c o", c=self.option_channels, o=self.option_dim), self.next_values[self.pos-1], value)

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size: Optional[int] = None) -> Generator[PPORolloutBufferSamples, None, None]:
        assert self.full, "Rollout buffer must be full before sampling from it"
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        indices = th.tensor(indices)

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            indices_batch = indices[start_idx : start_idx + batch_size]
            batch_inds = th.div(indices_batch, self.n_envs, rounding_mode="floor")
            env_inds = indices_batch % self.n_envs
            yield self._get_samples(batch_inds, env_inds)
            start_idx += batch_size

    def _get_samples(self, batch_inds: th.Tensor, env_inds: th.Tensor, env: Optional[VecNormalize] = None) -> PPORolloutBufferSamples:
        if isinstance(self.obs_shape, dict):
            observations = {key: self.to_device(obs[batch_inds, env_inds]) for (key, obs) in self.observations.items()}
            next_observations = {key: self.to_device(obs[batch_inds, env_inds]) for (key, obs) in self.next_observations.items()}
        else:
            observations = self.to_device(self.observations[batch_inds, env_inds])
            next_observations = self.to_device(self.next_observations[batch_inds, env_inds])

        def select(x):
            return self.to_device(x[batch_inds, env_inds])

        return PPORolloutBufferSamples(
            observations=observations,
            next_observations=next_observations,
            actions=select(self.actions),
            old_values=select(self.values),
            old_log_prob=select(self.log_probs),
            old_policy_matrix=select(self.policy_matrix),
            old_curr_next_options=select(self.curr_next_options),
            returns=select(self.returns),
            action_forward=select(self.action_forward),
            option_forward=select(self.option_forward),
            option_advantages=select(self.option_advantages),
            option_joint_probs=select(self.option_joint_probs),
        )

    def to_device(self, array: th.Tensor, copy: bool = True) -> th.Tensor:
        """
        :param array:
        :param copy: Whether to copy or not the data (may be useful to avoid changing things
            by reference). This argument is inoperative if the device is not the CPU.
        :return:
        """
        if copy:
            return array.clone().detach().to(self.device)
        return th.as_tensor(array, device=self.device)
