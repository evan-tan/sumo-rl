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
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch
import torch as T
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

    def forward(self, features: T.Tensor) -> Tuple[T.Tensor, T.Tensor]:
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

    def forward(self, features: T.Tensor) -> Tuple[T.Tensor, T.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        feat_ = self.fc(features)
        return self.policy_net(feat_), self.value_net(feat_)

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
        self.mlp_extractor = CustomNetwork(self.features_dim)

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.utils.annotations import override


# NOTE: For RLlib ONLY
class TorchRNNModel(TorchRNN, nn.Module):
    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        fc_size=64,
        lstm_state_size=256,
    ):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        self.obs_size = get_preprocessor(obs_space)(obs_space).size
        self.fc_size = fc_size
        self.lstm_state_size = lstm_state_size

        # Build the Module from fc + LSTM + 2xfc (action + value outs).
        self.fc1 = nn.Linear(self.obs_size, self.fc_size)
        self.lstm = nn.LSTM(self.fc_size, self.lstm_state_size, batch_first=True)
        self.action_branch = nn.Linear(self.lstm_state_size, num_outputs)
        self.value_branch = nn.Linear(self.lstm_state_size, 1)
        # Holds the current "base" output (before logits layer).
        self._features = None

    @override(ModelV2)
    def get_initial_state(self):
        # TODO: (sven): Get rid of `get_initial_state` once Trajectory
        #  View API is supported across all of RLlib.
        # Place hidden states on same device as model.
        h = [
            self.fc1.weight.new(1, self.lstm_state_size).zero_().squeeze(0),
            self.fc1.weight.new(1, self.lstm_state_size).zero_().squeeze(0),
        ]
        return h

    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return torch.reshape(self.value_branch(self._features), [-1])

    @override(TorchRNN)
    def forward_rnn(self, inputs, state, seq_lens):
        """Feeds `inputs` (B x T x ..) through the Gru Unit.
        Returns the resulting outputs as a sequence (B x T x ...).
        Values are stored in self._cur_value in simple (B) shape (where B
        contains both the B and T dims!).
        Returns:
            NN Outputs (B x T x ...) as sequence.
            The state batches as a List of two items (c- and h-states).
        """
        x = nn.functional.leaky_relu(self.fc1(inputs))
        self._features, [h, c] = self.lstm(
            x, [torch.unsqueeze(state[0], 0), torch.unsqueeze(state[1], 0)]
        )
        action_out = self.action_branch(self._features)
        return action_out, [torch.squeeze(h, 0), torch.squeeze(c, 0)]



if __name__ == "__main__":
    model = PPO(CustomActorCriticPolicy, "CartPole-v1", verbose=1)
    model.learn(5000)
