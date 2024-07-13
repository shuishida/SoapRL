from typing import Union, Optional, Callable, Dict, Any

import gymnasium as gym
import numpy as np
import torch as th

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import Image, Video

from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv


def rollout_policy(
    model: "type_aliases.PolicyPredictor",
    env: Union[gym.Env, VecEnv],
    deterministic: bool = True,
    n_eval_episodes: int = 1,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None
):
    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])

    n_envs = env.num_envs
    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    while (episode_counts < episode_count_targets).any():
        if callback is not None:
            callback(locals(), globals())

        actions, states = model.predict(observations, state=states, episode_start=episode_starts, deterministic=deterministic)
        observations, rewards, dones, infos = env.step(actions)
        episode_starts = dones

        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i] and dones[i]:
                episode_counts[i] += 1

        if render:
            env.render()


class VideoRecorderCallback(BaseCallback):
    def __init__(self, eval_env, render_freq: int, n_eval_episodes: int = 6, deterministic: bool = False):
        """
        Records a video of an agent's trajectory traversing ``eval_env`` and logs it to TensorBoard

        :param eval_env: A gym environment from which the trajectory is recorded
        :param render_freq: Render the agent's trajectory every eval_freq call of the callback.
        :param n_eval_episodes: Number of episodes to render
        :param deterministic: Whether to use deterministic or stochastic policy
        """
        super().__init__()
        self._eval_env = eval_env
        self._render_freq = render_freq
        self._n_eval_episodes = n_eval_episodes
        self._deterministic = deterministic

    def _on_step(self) -> bool:
        if self._render_freq > 0 and self.n_calls % self._render_freq == 0:
            self._save_video(self._eval_env, "rollout")
        return True

    def _save_video(self, eval_env, tag):
        screens = []

        def callback(_locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
            """
            Renders the environment in its current state, recording the screen in the captured `screens` list

            :param _locals: A dictionary containing all local variables of the callback's scope
            :param _globals: A dictionary containing all global variables of the callback's scope
            """
            screen = eval_env.render()
            # PyTorch uses CxHxW vs HxWxC gym (and tensorflow) image convention
            screens.append(screen.transpose(2, 0, 1))

        rollout_policy(
            self.model,
            eval_env,
            callback=callback,
            n_eval_episodes=self._n_eval_episodes,
            deterministic=self._deterministic,
        )

        self.logger.record(f"{tag}/image", Image(screens[-1], "CHW"), exclude=("stdout", "log", "json", "csv"))
        self.logger.record(
            f"{tag}/video",
            Video(th.ByteTensor(np.array([screens])), fps=1),
            exclude=("stdout", "log", "json", "csv"),
        )
