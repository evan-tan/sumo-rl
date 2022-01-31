from collections import OrderedDict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

torch.backends.cudnn.benchmark = False


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs,
    ):

        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False

        if "shared" in kwargs.keys():
            self.__shared = kwargs.get("shared")

    def _build_mlp_extractor(self) -> None:
        if self.__shared:
            self.mlp_extractor = CustomNetworkShared(self.features_dim)
        elif not self.__shared:
            self.mlp_extractor = CustomNetwork(self.features_dim)


class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.
    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 256,
    ):
        super(CustomNetwork, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.features_dim = feature_dim
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_pi),
            nn.Tanh(),
            nn.Linear(last_layer_dim_pi, last_layer_dim_pi),
            nn.Tanh(),
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_vf),
            nn.Tanh(),
            nn.Linear(last_layer_dim_vf, last_layer_dim_vf),
            nn.Tanh(),
        )

    def forward_actor(self, features):
        return self.policy_net(features)

    def forward_critic(self, features):
        return self.value_net(features)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.policy_net(features), self.value_net(features)


class CustomNetworkShared(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.
    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 256,
    ):
        super(CustomNetworkShared, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.features_dim = feature_dim
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        shared_dim = 256
        self.fc = nn.Linear(feature_dim, shared_dim)

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(shared_dim, last_layer_dim_pi),
            nn.Tanh(),
            nn.Linear(last_layer_dim_pi, last_layer_dim_pi),
            nn.Tanh(),
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(shared_dim, last_layer_dim_vf),
            nn.Tanh(),
            nn.Linear(last_layer_dim_vf, last_layer_dim_vf),
            nn.Tanh(),
        )

    def forward_actor(self, features):
        feat_ = self.fc(features)
        return self.policy_net(feat_)

    def forward_critic(self, features):
        feat_ = self.fc(features)
        return self.value_net(feat_)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        feat_ = self.fc(features)
        return self.policy_net(feat_), self.value_net(feat_)
