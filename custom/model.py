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
from torch import nn, optim
from torch.optim import lr_scheduler

torch.backends.cudnn.benchmark = False

class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param actor_out_dim: (int) number of units for the last layer of the policy network    
    :param critic_out_dim: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        actor_out_dim: int = 64,
        critic_out_dim: int = 64,
    ):
        super(CustomNetwork, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = actor_out_dim
        self.latent_dim_vf = critic_out_dim

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(256, actor_out_dim), nn.LeakyReLU()
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(256, critic_out_dim), nn.LeakyReLU()
        )

        self._num_layers = 1  # no stacked GRUs
        self._hidden_size = 256
        self.gru = nn.GRU(
            feature_dim,
            self._hidden_size,
            self._num_layers,
            batch_first=True,
            dropout=0,
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: (torch.Tensor, torch.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        batch_size = features.size(0)
        h_init = torch.zeros(self._num_layers, batch_size, self._hidden_size)

        # gru input shape: (batch size, seq_len, num_actions)
        out, h = self.gru(features, h_init)

        out = nn.LeakyReLU(out[:, -1, :])

        # share input for actor critic
        return self.policy_net(out), self.value_net(out)


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

    def _build_mlp_extractor(self) -> None:
        # this is actually a GRU...
        self.mlp_extractor = CustomNetwork(self.features_dim)


if __name__ == "__main__":
    model = PPO(CustomActorCriticPolicy, "CartPole-v1", verbose=1)
    model.learn(5000)
