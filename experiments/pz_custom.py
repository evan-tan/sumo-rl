import argparse
import json
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import ray
import ray.rllib.agents.ppo as ppo
import sumo_rl
import supersuit as ss
import tensorflow as tf
import torch
import torch.nn as nn
from array2gif import write_gif
from custom.model import CustomActorCriticPolicy
from custom.utils import load_cfg
from gym.spaces import Box
from numpy import float32
from ray import tune
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.env import PettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.tf.recurrent_net import RecurrentNetwork
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.torch_ops import FLOAT_MAX, FLOAT_MIN
from ray.tune.registry import register_env


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
        x = nn.functional.relu(self.fc1(inputs))
        self._features, [h, c] = self.lstm(
            x, [torch.unsqueeze(state[0], 0), torch.unsqueeze(state[1], 0)]
        )
        action_out = self.action_branch(self._features)
        return action_out, [torch.squeeze(h, 0), torch.squeeze(c, 0)]


def env_creator(num_timesteps):
    """For PettingZoo"""
    env = sumo_rl.env(
        net_file="nets/big-intersection/big-intersection.net.xml",
        route_file="nets/big-intersection/routes.rou.xml",
        out_csv_name="outputs/big-intersection/test",
        use_gui=False,
        num_seconds=int(num_timesteps),
    )
    return env


if __name__ == "__main__":
    n_timesteps = int(2.5e6)
    algo_name = "PPO".upper()
    model_name = "lstm"
    env_name = "sumo_pz"
    num_cpus = 6
    num_rollouts = 12

    ModelCatalog.register_custom_model(model_name, TorchRNNModel)
    register_env(env_name, lambda config: PettingZooEnv(env_creator(n_timesteps)))

    tmp_env = PettingZooEnv(env_creator(n_timesteps))
    policy_dict = {
        "policy_0": (None, tmp_env.observation_space, tmp_env.action_space, {})
    }
    policy_ids = list(policy_dict.keys())
    assert len(policy_ids) == 1

    config = deepcopy(get_agent_class(algo_name)._default_config)
    # https://docs.ray.io/en/latest/rllib-training.html#common-parameters
    # https://docs.ray.io/en/latest/rllib-algorithms.html#proximal-policy-optimization-ppo
    config["multiagent"] = {
        "policies": policy_dict,
        "policy_mapping_fn": (lambda agent_id: policy_ids[0]),
    }

    config["framework"] = "torch"
    config["log_level"] = "DEBUG"
    config["num_workers"] = 1
    config["num_gpus"] = int(os.environ.get("RLLIB_NUM_GPUS", "0"))
    config["rollout_fragment_length"] = 30
    # Training batch size, if applicable. Should be >= rollout_fragment_length.
    # Samples batches will be concatenated together to a batch of this size,
    # which is then passed to SGD.
    config["train_batch_size"] = 200
    config["horizon"] = 200  # after n steps, reset sim
    config["no_done_at_end"] = False
    config["model"] = {
        "custom_model": model_name,
    }
    # TODO: doesn't seem to work, how to eval?
    # config["evaluation_interval"] = "auto"
    # config["evaluation_parallel_to_training"] = True

    # config['hiddens'] = []
    # you must first register env_name so that RLlib knows it exists
    config["env"] = env_name

    ray.init(num_cpus=num_cpus)

    run_name = algo_name + "-" + env_name
    cwd = str(Path.cwd() / "ray_results")
    tune.run(
        algo_name,
        name=run_name,
        stop={"timesteps_total": n_timesteps},
        checkpoint_freq=100,
        config=config,
        local_dir=cwd,
    )
