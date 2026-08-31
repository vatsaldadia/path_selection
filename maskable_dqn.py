from stable_baselines3 import DQN
from maskable_multi_input import MaskableMultiInputPolicy
import gymnasium as gym
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.utils import explained_variance, get_linear_fn, is_vectorized_observation, safe_mean
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
import warnings
import numpy as np
import torch as th
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Callable
from gymnasium import spaces

class MaskableDQN(DQN):
    """
    DQN with support for MultiInput observations and action masking.

    :param policy: The policy model to use (e.g., "MlpPolicy")
    :param env: The environment to learn from (can be None for loading a trained model)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: Number of steps before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: Discount factor
    :param train_freq: Update the model every ``train_freq`` steps.
    :param gradient_steps: How many gradient steps to do after each rollout
        (see ``train_freq`` and ``n_episodes_rollout``)
        Set to -1 means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param replay_buffer_class: Class of the replay buffer
        (defaults to ReplayBuffer)
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See [https://stable-baselines3.readthedocs.io/en/master/guide/buffers.html](https://stable-baselines3.readthedocs.io/en/master/guide/buffers.html)
    :param target_update_interval: update the target network every ``target_update_interval``
        environment steps.
    :param exploration_fraction: Fraction of entire training period over which the exploration rate is reduced
    :param exploration_initial_eps: Initial value of random action probability
    :param exploration_final_eps: Final value of random action probability
    :param max_grad_norm: The maximum value for the gradient clipping
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (training progress), 2 for debug debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, cuda:1, ...) on which the code should be run.
        When None, it will choose the most suitable device (gpu if present and available, otherwise cpu)
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """
    def __init__(
        self,
        policy: Union[str, Type[MaskableMultiInputPolicy]], # Type hint for our custom policy
        env: Union[gym.Env, VecEnv, str],
        learning_rate: Union[float, Callable[[float], float]] = 1e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 50000,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = (4, "step"),
        gradient_steps: int = 1,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            target_update_interval=target_update_interval,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            max_grad_norm=max_grad_norm,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )

    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample an action according to the exploration policy.
        This is either done by sampling the probability distribution of the policy,
        or sampling a random action (from a uniform distribution over the action space)
        or by adding noise to the deterministic output.

        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param n_envs:
        :return: action to take in the environment
            and scaled action that will be stored in the replay buffer.
            The two differs when the action space is not normalized (bounds are not [-1, 1]).
        """
        action_mask = self._last_obs["action_mask"]
        # Select action randomly or according to policy
        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            # Warmup phase
            unscaled_action = np.array([self.action_space.sample(mask=action_mask[i]) for i in range(n_envs)])
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
            assert self._last_obs is not None, "self._last_obs was not set"
            unscaled_action, _ = self.predict(self._last_obs, action_masks=action_mask, deterministic=False)

        return unscaled_action, unscaled_action

    # noinspection method-overriding
    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        action_masks: Optional[np.ndarray] = None,
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Overrides the base_class predict function to include epsilon-greedy exploration.

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :param action_masks: action mask of current step
        :return: the model's action and the next flow
            (used in recurrent policies)
        """
        if not deterministic and np.random.rand() < self.exploration_rate:
            if self.policy.is_vectorized_observation(observation):
                if isinstance(observation, dict):
                    n_batch = observation[next(iter(observation.keys()))].shape[0]
                else:
                    n_batch = observation.shape[0]
                action = np.array([self.action_space.sample(mask=action_masks[i]) for i in range(n_batch)])
            else:
                action = np.array(self.action_space.sample(mask=action_masks))
        else:
            action, state = self.policy.predict(observation, state, episode_start, deterministic)
        return action, state