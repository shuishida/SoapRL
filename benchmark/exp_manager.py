import os
from pprint import pprint
from typing import Any, Dict, Optional, Tuple

import optuna
from rl_zoo3.exp_manager import ExperimentManager
from sb3_contrib.common.vec_env import AsyncEval

# For using HER with GoalEnv
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike  # noqa: F401
from stable_baselines3.common.vec_env import (
    VecEnv,
)

# Register custom envs
import rl_zoo3.import_envs  # noqa: F401
from rl_zoo3.callbacks import TrialEvalCallback
from rl_zoo3.hyperparams_opt import HYPERPARAMS_SAMPLER
from rl_zoo3.utils import get_callback_list

import sys
sys.path.append(".")
from benchmark.utils import ALGOS


class MyExperimentManager(ExperimentManager):
    """
    Experiment manager: read the hyperparameters,
    preprocess them, create the environment and the RL model.

    Please take a look at `train.py` to have the details for each argument.
    """
    def setup_experiment(self) -> Optional[Tuple[BaseAlgorithm, Dict[str, Any]]]:
        """
        Read hyperparameters, pre-process them (create schedules, wrappers, callbacks, action noise objects)
        create the environment and possibly the model.

        :return: the initialized RL model
        """
        unprocessed_hyperparams, saved_hyperparams = self.read_hyperparameters()
        hyperparams, self.env_wrapper, self.callbacks, self.vec_env_wrapper = self._preprocess_hyperparams(
            unprocessed_hyperparams
        )

        self.create_log_folder()
        self.create_callbacks()

        # Create env to have access to action space for action noise
        n_envs = 1 if self.algo == "ars" or self.optimize_hyperparameters else self.n_envs
        env = self.create_envs(n_envs, no_log=False)

        self._hyperparams = self._preprocess_action_noise(hyperparams, saved_hyperparams, env)

        if self.continue_training:
            model = self._load_pretrained_agent(self._hyperparams, env)
        elif self.optimize_hyperparameters:
            env.close()
            return None
        else:
            # Train an agent from scratch
            model = ALGOS[self.algo](
                env=env,
                tensorboard_log=self.tensorboard_log,
                seed=self.seed,
                verbose=self.verbose,
                device=self.device,
                **self._hyperparams,
            )

        self._save_config(saved_hyperparams)
        return model, saved_hyperparams

    def _load_pretrained_agent(self, hyperparams: Dict[str, Any], env: VecEnv) -> BaseAlgorithm:
        # Continue training
        print("Loading pretrained agent")
        # Policy should not be changed
        del hyperparams["policy"]

        if "policy_kwargs" in hyperparams.keys():
            del hyperparams["policy_kwargs"]

        model = ALGOS[self.algo].load(
            self.trained_agent,
            env=env,
            seed=self.seed,
            tensorboard_log=self.tensorboard_log,
            verbose=self.verbose,
            device=self.device,
            **hyperparams,
        )

        replay_buffer_path = os.path.join(os.path.dirname(self.trained_agent), "replay_buffer.pkl")

        if os.path.exists(replay_buffer_path):
            print("Loading replay buffer")
            # `truncate_last_traj` will be taken into account only if we use HER replay buffer
            assert hasattr(
                model, "load_replay_buffer"
            ), "The current model doesn't have a `load_replay_buffer` to load the replay buffer"
            model.load_replay_buffer(replay_buffer_path, truncate_last_traj=self.truncate_last_trajectory)
        return model

    def objective(self, trial: optuna.Trial) -> float:
        kwargs = self._hyperparams.copy()

        n_envs = 1 if self.algo == "ars" else self.n_envs

        additional_args = {
            "using_her_replay_buffer": kwargs.get("replay_buffer_class") == HerReplayBuffer,
            "her_kwargs": kwargs.get("replay_buffer_kwargs", {}),
        }
        # Pass n_actions to initialize DDPG/TD3 noise sampler
        # Sample candidate hyperparameters
        sampled_hyperparams = HYPERPARAMS_SAMPLER[self.algo](trial, self.n_actions, n_envs, additional_args)
        kwargs.update(sampled_hyperparams)

        env = self.create_envs(n_envs, no_log=True)

        # By default, do not activate verbose output to keep
        # stdout clean with only the trial's results
        trial_verbosity = 0
        # Activate verbose mode for the trial in debug mode
        # See PR #214
        if self.verbose >= 2:
            trial_verbosity = self.verbose

        model = ALGOS[self.algo](
            env=env,
            tensorboard_log=None,
            # We do not seed the trial
            seed=None,
            verbose=trial_verbosity,
            device=self.device,
            **kwargs,
        )

        eval_env = self.create_envs(n_envs=self.n_eval_envs, eval_env=True)

        optuna_eval_freq = int(self.n_timesteps / self.n_evaluations)
        # Account for parallel envs
        optuna_eval_freq = max(optuna_eval_freq // self.n_envs, 1)
        # Use non-deterministic eval for Atari
        path = None
        if self.optimization_log_path is not None:
            path = os.path.join(self.optimization_log_path, f"trial_{trial.number!s}")
        callbacks = get_callback_list({"callback": self.specified_callbacks})
        eval_callback = TrialEvalCallback(
            eval_env,
            trial,
            best_model_save_path=path,
            log_path=path,
            n_eval_episodes=self.n_eval_episodes,
            eval_freq=optuna_eval_freq,
            deterministic=self.deterministic_eval,
        )
        callbacks.append(eval_callback)

        learn_kwargs = {}
        # Special case for ARS
        if self.algo == "ars" and self.n_envs > 1:
            learn_kwargs["async_eval"] = AsyncEval(
                [lambda: self.create_envs(n_envs=1, no_log=True) for _ in range(self.n_envs)], model.policy
            )

        try:
            model.learn(self.n_timesteps, callback=callbacks, **learn_kwargs)  # type: ignore[arg-type]
            # Free memory
            assert model.env is not None
            model.env.close()
            eval_env.close()
        except (AssertionError, ValueError) as e:
            # Sometimes, random hyperparams can generate NaN
            # Free memory
            assert model.env is not None
            model.env.close()
            eval_env.close()
            # Prune hyperparams that generate NaNs
            print(e)
            print("============")
            print("Sampled hyperparams:")
            pprint(sampled_hyperparams)
            raise optuna.exceptions.TrialPruned() from e
        is_pruned = eval_callback.is_pruned
        reward = eval_callback.last_mean_reward

        del model.env, eval_env
        del model

        if is_pruned:
            raise optuna.exceptions.TrialPruned()

        return reward
