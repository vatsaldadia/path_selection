import gymnasium as gym
import torch as th
from torch import nn
from stable_baselines3.dqn.policies import QNetwork
from stable_baselines3.common.torch_layers import create_mlp
from typing import Any, Dict, List, Type, Optional
from stable_baselines3.common.type_aliases import PyTorchObs

from maskable_features_extractor import MaskableFeaturesExtractor
from rehearsal_buffer import RehearsalBuffer

class MaskableQNetwork(QNetwork):
    """
    A custom Q-network that adheres to the SB3 QNetwork API
    but uses a mask internally.
    """

    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Discrete,
            features_extractor: MaskableFeaturesExtractor,
            features_dim: int,
            net_arch: Optional[List[int]] = None,
            activation_fn: Type[nn.Module] = nn.ReLU,
            normalize_images: bool = True,
    ):

        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            features_extractor=features_extractor,
            features_dim=features_dim,
            net_arch=net_arch,
            activation_fn=activation_fn,
            normalize_images=normalize_images
        )

    def forward(self, obs: PyTorchObs) -> th.Tensor:
        """
        Overrides the parent's forward pass to handle our graph observation.
        """
        features = self.features_extractor(obs)
        return self.q_net(features)

    def _predict(self, observation: PyTorchObs, deterministic: bool = True) -> th.Tensor:

        """
        Predict the action based on the observation, applying the action mask if provided.
        :param observation: The observation from the environment
        :param deterministic: Whether to use a deterministic policy
        :return: The predicted action
        """
        q_values = self(observation)

        if isinstance(observation, dict) and "action_mask" in observation:
            action_mask = observation["action_mask"].bool()
            q_values[~action_mask] = -1e9  # Set q-values of invalid actions to -infinity

        action = q_values.argmax(dim=1).reshape(-1)
        return action