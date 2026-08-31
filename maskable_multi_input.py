from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.dqn.policies import QNetwork

from maskable_features_extractor import MaskableFeaturesExtractor
from rehearsal_buffer import RehearsalBuffer
from maskable_qnetwork import MaskableQNetwork
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
)
from stable_baselines3.common.type_aliases import Schedule

import torch.nn as nn
import torch as th
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from gymnasium import spaces


class MaskableMultiInputPolicy(DQNPolicy):

    """
    Policy class with Q-Value Net and target net for DQN

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features' extractor.
    :param normalize_images: Whether to normalise images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimiser to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimiser
    """

    q_net: MaskableQNetwork
    q_net_target: MaskableQNetwork

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        net_arch: Optional[list[int]] = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        features_extractor_class: type[BaseFeaturesExtractor] = MaskableFeaturesExtractor,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            normalize_images=normalize_images,
        )

    def make_q_net(self) -> MaskableQNetwork:
        # noinspection PyTypeChecker
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)      # make sure we always have separate networks for features extractors etc.
        return MaskableQNetwork(**net_args).to(self.device)