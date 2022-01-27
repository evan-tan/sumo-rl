import datetime

import gym
import matplotlib.pyplot as plt
import numpy as np
import psutil
import sumo_rl
import supersuit as ss
import torch
from custom import utils
from custom.sb3_model import CustomActorCriticPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

if __name__ == "__main__":

    n_evaluations = 20
    cfg = utils.load_cfg("../config.yml")

    # NOTE: You can not use LIBSUMO if using more than one env
    assert cfg.num_cpus < psutil.cpu_count()
    assert cfg.num_envs < cfg.num_cpus

    total_timesteps = cfg.train.num_episodes * cfg.train.env_timeout * cfg.num_envs

    # eval after every episode
    eval_freq = cfg.train.num_episodes // cfg.env_tstep

    env = utils.env_creator(cfg, "train")
    eval_env = utils.env_creator(cfg, "eval")

    # define network architecture
    policy_kwargs = dict(
        activation_fn=torch.nn.Tanh, net_arch=[256, dict(vf=[256, 256], pi=[64, 64])]
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=3,
        gamma=0.99,
        n_steps=256,
        ent_coef=0.0905168,
        learning_rate=utils.linear_schedule(1e-4),
        vf_coef=0.042202,
        max_grad_norm=0.9,
        gae_lambda=0.9,
        n_epochs=5,
        clip_range=0.25,
        batch_size=256,
        policy_kwargs=policy_kwargs,
        tensorboard_log=cfg.output_path + "tensorboard",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=cfg.output_path + "best_model",
        log_path=cfg.output_path + "eval",
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        n_eval_episodes=1,
    )

    train = True
    if train:
        start_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        print(f"Start @ {start_time}")

        model.learn(total_timesteps=total_timesteps, callback=eval_callback)
        end_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        print(f"End @ {end_time}")

        # save a learned model
        model.save(cfg.output_path + "final_model")
