# %%
import argparse
import json
import os
from copy import deepcopy
from pathlib import Path

import gym
import numpy as np
import pickle5 as pickle
import psutil
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
from gym.envs.registration import register
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

# from ray.tune.registry import register


def unsqueeze(arr: np.ndarray, dim: int) -> np.ndarray:
    """Wrapper function for torch.unsqueeze() functionality in NumPy"""
    return np.expand_dims(arr, axis=dim)


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


def env_creator(num_timesteps, parallel=False):
    """For PettingZoo"""
    env_timestep = 5
    if parallel:
        env = sumo_rl.parallel_env(
            net_file="nets/big-intersection/big-intersection.net.xml",
            route_file="nets/big-intersection/routes.rou.xml",
            out_csv_name="outputs/big-intersection/test",
            use_gui=True,
            num_seconds=int(num_timesteps),
        )
    else:
        env = sumo_rl.env(
            net_file="nets/big-intersection/big-intersection.net.xml",
            route_file="nets/big-intersection/routes.rou.xml",
            out_csv_name="outputs/big-intersection/test",
            use_gui=True,
            num_seconds=int(num_timesteps),
        )

    return env


# ppo configuration
#    env,
#     verbose=3,
#     gamma=0.95,
#     n_steps=256,
#     ent_coef=0.0905168,
#     learning_rate=0.00062211,
#     vf_coef=0.042202,
#     max_grad_norm=0.9,
#     gae_lambda=0.99,
#     n_epochs=5,
#     clip_range=0.3,
#     batch_size=256,
# )


# TODO: fix path
# register OpenAI Gym environment
# register(
#     id="sumo_rl_v0",
#     entry_point="sumo_rl.environment:env",
# )
# TODO:
# custom class
# methods: load model, save model, select checkpoint, training, testing


class CustomHandler:
    def __init__(self, cfg):
        self._env = None
        self._model = TorchRNNModel()

    def train(self):
        pass

    def test(self):
        pass

    def save(self):
        pass

    def load_checkpoint(self):
        pass


if __name__ == "__main__":
    # 3070 GPU, 7.5 hours... 1 worker...
    # 4x4 -> 2.5e6 steps
    # big intersection -> 5.4e5 steps
    num_workers = 0
    num_timesteps = int(5.4e5 * num_workers)
    algo_name = "PPO".upper()
    model_name = "lstm"
    env_name = "sumo_pz"
    num_cpus = int(psutil.cpu_count() - 2)

    # # sumorl-gui-v0
    # env = gym.make("sumorl-v0")
    env = env_creator(num_timesteps, parallel=False)

    ModelCatalog.register_custom_model(model_name, TorchRNNModel)
    register_env(env_name, lambda config: PettingZooEnv(env_creator(num_timesteps)))

    num_epochs = 10
    config = deepcopy(get_agent_class(algo_name)._default_config)
    config["env"] = env_name
    config["action_space"] = env.action_space
    config["observation_space"] = env.observation_space
    config["num_workers"] = 0
    config["num_gpus"] = 0
    config["in_evaluation"] = True
    config["evaluation_num_workers"] = 1
    # config['custom_eval_function'] = ...
    config["evaluation_num_episodes"] = 2000
    config["evaluation_interval"] = 2000
    eval_cfg = {
        "explore": False,
        "env_config": {
            # Use test set to evaluate
            "mode": "test"
        },
    }
    config["evaluation_config"] = eval_cfg

    # agent = ppo.PPOTrainer(config=config, env=env_name)
    checkpoint_path = "ray_results/PPO-sumo_pz/PPO_sumo_pz_a2aca_00000_0_2021-10-19_13-37-23/checkpoint_001900/checkpoint-1900"
    params_path = Path(checkpoint_path).parents[1] / "params.pkl"

    # config["multiagent"] = {
    #     "policies": policy_dict,
    #     "policy_mapping_fn": (lambda agent_id: policy_ids[0]),
    # }
    del config
    with open(params_path, "rb") as f:
        config = pickle.load(f)
        del config["_disable_preprocessor_api"]
        del config["num_workers"]
        del config["num_gpus"]
        ray.init(num_cpus=num_cpus)
        PPOAgent = ppo.PPOTrainer(config=config, env=env_name)
        PPOAgent.restore(checkpoint_path)

        done = False
        env.reset()
        # single agent
        reward_sums = {a: 0 for a in env.possible_agents}

        cell_size = 256
        state = [np.zeros(cell_size, np.float32), np.zeros(cell_size, np.float32)]

        for agent in env.agent_iter():
            # agent == "TL"
            assert agent in reward_sums.keys()

            obs, reward, done, info = env.last()
            # access observation from dict
            # convert to obs_dict
            obs = {"obs": obs, "obs_flat": obs.flatten()}

            reward_sums[agent] += reward
            if done:
                action = None
            else:
                # mismatched agent names
                # agent == "TL" but policy named "policy_0"
                policy = PPOAgent.get_policy("policy_0")
                # action, state, logits = policy.compute_action(obs, state)
                batch_action, state_out, info = policy.compute_actions_from_input_dict(
                    obs, state
                )
                # single_action = batch_action[0]
                # action = single_action

                # env.step(action)
                # print("Rewards: ", reward_sums)

# tmp_env = PettingZooEnv(env_creator(num_timesteps))
# policy_dict = {
#     "policy_0": (None, tmp_env.observation_space, tmp_env.action_space, {})
# }
# policy_ids = list(policy_dict.keys())
# assert len(policy_ids) == 1

# config = deepcopy(get_agent_class(algo_name)._default_config)
# # https://docs.ray.io/en/latest/rllib-training.html#common-parameters
# # https://docs.ray.io/en/latest/rllib-algorithms.html#proximal-policy-optimization-ppo
# config["multiagent"] = {
#     "policies": policy_dict,
#     "policy_mapping_fn": (lambda agent_id: policy_ids[0]),
# }

# config["framework"] = "torch"
# config["log_level"] = "DEBUG"
# config["num_workers"] = num_workers
# config["num_gpus"] = 1 if torch.cuda.is_available() else 0
# config["rollout_fragment_length"] = 30
# # Training batch size, if applicable. Should be >= rollout_fragment_length.
# # Samples batches will be concatenated together to a batch of this size,
# # which is then passed to SGD.
# config["train_batch_size"] = 200 * num_workers
# # after n steps, reset sim,
# # NOTE: this shoudl match max_steps // 5 in TrafficGenerator
# config["horizon"] = 8000
# config["no_done_at_end"] = False
# config["model"] = {
#     "custom_model": model_name,
# }
# config["lr"] = 5e-5  # default: 5e-5
# config["lr_schedule"] = None  # default: None
# config["sgd_minibatch_size"] = 128  # default: 128
# # TODO: doesn't seem to work, how to eval?
# # config["evaluation_interval"] = "auto"
# # config["evaluation_parallel_to_training"] = True

# # config['hiddens'] = []
# # you must first register env_name so that RLlib knows it exists
# config["env"] = env_name

# ray.init(num_cpus=num_cpus)

# run_name = algo_name + "-" + env_name
# cwd = str(Path.cwd() / "ray_results")
# tune.run(
#     algo_name,
#     name=run_name,
#     stop={"timesteps_total": num_timesteps},
#     checkpoint_freq=100,
#     config=config,
#     local_dir=cwd,
# )
