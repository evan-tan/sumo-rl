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
from custom.model import CustomActorCriticPolicy, TorchRNNModel
from custom.utils import env_creator, load_cfg, unsqueeze
from gym.envs.registration import register
from gym.spaces import Box
from numpy import float32
from ray import tune
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.env import PettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env


class CustomHandler:
    """How to use...
    Instantiate (automatically calls configure and )
    """

    def __init__(self):
        # 3070 GPU, 7.5 hours... 1 worker...
        # 4x4 -> 2.5e6 steps
        # big intersection -> 5.4e5 steps
        self.algo_name = "PPO".upper()
        self.default_cfg = deepcopy(get_agent_class(self.algo_name)._default_config)
        self.env_name = "sumo_pz"
        self._model_class = TorchRNNModel
        self._num_cpus = int(psutil.cpu_count() - 2)
        self._max_env_timesteps = None
        self._cwd = Path.cwd()
        self.paths = {
            "net": str(
                self._cwd / "nets/big-intersection/big-intersection.net.xml"
            ),
            "route": str(self._cwd / "nets/big-intersection/routes.rou.xml"),
            "output_csv": str(self._cwd / "outputs/big-intersection/test"),
        }

    def register_envs(self):
        assert self._max_env_timesteps is not None

        # you must first register model, or env so that RLlib knows it exists
        ModelCatalog.register_custom_model(
            self._model_class.__name__, self._model_class
        )

        register_env(
            self.env_name,
            lambda config: PettingZooEnv(env_creator(self._max_env_timesteps, self.paths)),
        )

    def configure(self, mode):
        """Configure parameters based on train or test"""
        if mode == "test":
            self._max_env_timesteps = int(5.4e5)
            self.env = env_creator(self._max_env_timesteps, self.paths)

            self.test_cfg = deepcopy(self.default_cfg, self.paths)
            self.test_cfg["env"] = self.env_name
            tmp_env = env_creator(self._max_env_timesteps, self.paths)
            self.test_cfg["action_space"] = tmp_env.action_space
            self.test_cfg["observation_space"] = tmp_env.observation_space
            self.test_cfg["num_workers"] = 0
            self.test_cfg["num_gpus"] = 0
            self.test_cfg["in_evaluation"] = True
            self.test_cfg["evaluation_num_workers"] = 1
            # config['custom_eval_function'] = ...
            self.test_cfg["evaluation_num_episodes"] = 2000
            self.test_cfg["evaluation_interval"] = 2000
            eval_cfg = {
                "explore": False,
                "env_config": {"mode": "test"},
            }
            self.test_cfg["evaluation_config"] = eval_cfg
        elif mode == "train":
            self.train_cfg = deepcopy(self.default_cfg)
            # https://docs.ray.io/en/latest/rllib-training.html#common-parameters
            # https://docs.ray.io/en/latest/rllib-algorithms.html#proximal-policy-optimization-ppo

            num_train_workers = int(psutil.cpu_count() - 8)
            assert num_train_workers <= self._num_cpus
            self._max_env_timesteps = int(5.4e5) * num_train_workers
            self.env = env_creator(self._max_env_timesteps, self.paths)

            self.train_cfg["env"] = self.env_name
            self.train_cfg["framework"] = "torch"
            self.train_cfg["log_level"] = "DEBUG"
            self.train_cfg["num_workers"] = num_train_workers
            self.train_cfg["num_gpus"] = 1 if torch.cuda.is_available() else 0
            self.train_cfg["num_sgd_iter"] =  30
            self.train_cfg["rollout_fragment_length"] = 100
            # Training batch size, if applicable. Should be >= rollout_fragment_length.
            # Samples batches will be concatenated together to a batch of this size,
            # which is then passed to SGD.
            # self.train_cfg["train_batch_size"] = 200 * num_train_workers
            self.train_cfg["train_batch_size"] = self.train_cfg["rollout_fragment_length"] * num_train_workers
            # after n steps, reset sim,
            # NOTE: this shoudl match max_steps // 5 in TrafficGenerator
            self.train_cfg["horizon"] = 8000
            self.train_cfg["no_done_at_end"] = False
            self.train_cfg["model"] = {
                "custom_model": self._model_class.__name__,
            }
            self.train_cfg["lr"] = 5e-5  # default: 5e-5
            self.train_cfg["lr_schedule"] = None  # default: None
            self.train_cfg["sgd_minibatch_size"] = 64  # default: 128

    def train(self):
        tmp_env = PettingZooEnv(self.env)
        policy_dict = {
            "TL": (None, tmp_env.observation_space, tmp_env.action_space, {})
        }
        policy_ids = list(policy_dict.keys())
        assert len(policy_ids) == 1
        self.train_cfg["multiagent"] = {
            "policies": policy_dict,
            "policy_mapping_fn": (lambda agent_id: policy_ids[0]),
        }

        ray.init(num_cpus=self._num_cpus)

        run_name = self.algo_name + "-" + self.env_name
        log_dir = str(self._cwd / "ray_results")
        tune.run(
            self.algo_name,
            name=run_name,
            stop={"timesteps_total": self._max_env_timesteps},
            checkpoint_freq=100,
            config=self.train_cfg,
            local_dir=log_dir,
        )

    def test(self, checkpoint_path):
        """Checkpoint path example:
        ray_results/PPO-sumo_pz/PPO_sumo_pz_a2aca_00000_0_2021-10-19_13-37-23/checkpoint_001900/checkpoint-1900 <- actual checkpoint file
        """
        params_path = Path(checkpoint_path).parents[1] / "params.pkl"
        with open(params_path, "rb") as f:
            self.test_cfg = pickle.load(f)
            del self.test_cfg["_disable_preprocessor_api"]
            del self.test_cfg["num_workers"]
            del self.test_cfg["num_gpus"]

            ray.init(num_cpus=self._num_cpus, num_gpus=0)
            PPOAgent = ppo.PPOTrainer(config=self.test_cfg, env=self.env_name)
            PPOAgent.restore(checkpoint_path)

            done = False
            reward_sums = {a: 0 for a in self.env.possible_agents}
            # TODO: infer from model
            cell_size = 256
            state = [
                unsqueeze(np.zeros(cell_size, np.float32), dim=0),
                unsqueeze(np.zeros(cell_size, np.float32), dim=0),
            ]

            self.env.reset()
            for agent in self.env.agent_iter():
                # agent == "TL"
                assert agent in reward_sums.keys()
                # print(agent)
                obs, reward, done, info = self.env.last()
                # access observation from dict, wrap into obs_dict
                obs = {"obs": obs, "obs_flat": obs.flatten()}
                obs["obs"] = unsqueeze(obs["obs"], dim=0)

                reward_sums["TL"] += reward
                if done:
                    action = None
                else:
                    # mismatched agent names
                    # agent == "TL" but policy named "policy_0"
                    policy = PPOAgent.get_policy("policy_0")
                    # action, state, logits = policy.compute_action(obs, state)
                    batch_action, state_out, info = policy.compute_actions(
                        obs["obs"], state
                    )
                    action = batch_action[0]

                self.env.step(action)
                print("Rewards: ", reward_sums)

    def save(self):
        pass

    def load_checkpoint(self):
        pass


if __name__ == "__main__":
    handler = CustomHandler()
    # train model
    handler.configure("train")
    handler.register_envs()
    handler.train()

    # save model

    # # test model
    # checkpoint_path = "ray_results/PPO-sumo_pz/PPO_sumo_pz_a2aca_00000_0_2021-10-19_13-37-23/checkpoint_001900/checkpoint-1900"
    # handler.configure("test")
    # handler.register_envs()
    # handler.test(checkpoint_path)
