import argparse
import json
import os
from copy import deepcopy

import numpy as np
import ray
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
from pettingzoo.classic import leduc_holdem_v2
from ray import tune
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.env import PettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.torch_ops import FLOAT_MAX, FLOAT_MIN
from ray.tune.registry import register_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecMonitor
from supersuit import dtype_v0

# tf1, tf, tfv = try_import_tf()
# torch, nn = try_import_torch()


class CustomNetwork(TorchModelV2, torch.nn.Module):
    """PyTorch version of above ParametricActionsModel."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kw):
        torch.nn.Module.__init__(self)
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name, **kw
        )

        obs_len = obs_space.shape[0] - action_space.n

        orig_obs_space = Box(
            shape=(obs_len,), low=obs_space.low[:obs_len], high=obs_space.high[:obs_len]
        )
        self.action_embed_model = TorchFC(
            orig_obs_space,
            action_space,
            action_space.n,
            model_config,
            name + "_action_embed",
        )

        self.actor = nn.Linear(512, 64)  # policy function
        self.critic = nn.Linear(512, 1)  # value function

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the predicted action embedding
        action_logits, _ = self.action_embed_model(
            {"obs": input_dict["obs"]["observation"]}
        )
        # turns probit action mask into logit action mask
        inf_mask = torch.clamp(torch.log(action_mask), -1e10, FLOAT_MAX)

        return action_logits + inf_mask, state

    def value_function(self):
        return self.action_embed_model.value_function()


def env_creator():
    """For PettingZoo"""
    env = sumo_rl.env(
        net_file="nets/4x4-Lucas/4x4.net.xml",
        route_file="nets/4x4-Lucas/4x4c1c2c1c2.rou.xml",
        out_csv_name="outputs/4x4grid/test",
        use_gui=False,
        num_seconds=int(1e3),
    )
    return env


if __name__ == "__main__":
    algo_name = "PPO".upper()
    model_name = "gru"
    env_name = "sumo_pz"
    num_cpus = 6
    num_rollouts = 12

    ModelCatalog.register_custom_model(model_name, CustomNetwork)
    register_env(env_name, lambda config: PettingZooEnv(env_creator()))
    config = deepcopy(get_agent_class(algo_name)._default_config)

    tmp_env = PettingZooEnv(env_creator())
    obs_space = tmp_env.observation_space
    act_space = tmp_env.action_space

    config["multiagent"] = {
        "policies": {
            "agent_1": (None, obs_space, act_space, {}),
        },
        "policy_mapping_fn": lambda agent_id: agent_id,
    }

    config["framework"] = "torch"
    config["log_level"] = "DEBUG"
    config["num_workers"] = 1
    config["num_gpus"] = int(os.environ.get("RLLIB_NUM_GPUS", "0"))
    config["rollout_fragment_length"] = 30
    config["train_batch_size"] = 200
    config["horizon"] = 200  # after n steps, reset sim
    config["no_done_at_end"] = False
    config["model"] = {
        "custom_model": model_name,
    }

    # config['hiddens'] = []
    config["env"] = env_name

    ray.init(num_cpus=num_cpus + 1)

    run_name = algo_name + "-" + env_name
    tune.run(
        algo_name,
        name=run_name,
        stop={"timesteps_total": 1000},
        checkpoint_freq=10,
        config=config,
    )
