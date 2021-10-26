import json
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import psutil
import sumo_rl
import supersuit as ss
import torch
from array2gif import write_gif
from custom.model import CustomActorCriticPolicy
from custom.utils import load_cfg, smooth_data
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import VecMonitor

# NOTE: Don't forget to execute this script from 1 directory above experiments/

if __name__ == "__main__":

    sumo_tstep = 7
    n_evaluations = 20
    num_cpus = int(psutil.cpu_count() - 1)
    # You can not use LIBSUMO if using more than one env
    num_envs = int(num_cpus - 4)
    # NOTE: these are actual sumo time steps!!
    # determine when to reset envs!!
    train_timeout = int(1.5e4)
    eval_timeout = int(2.5e3)
    num_episodes = 1000
    # actual number of time steps
    total_timesteps = num_episodes * train_timeout * num_envs
    # eval after every episode
    eval_freq = train_timeout // sumo_tstep
    save_path = "./logs/"

    env = sumo_rl.parallel_env(
        net_file="nets/big-intersection/big-intersection.net.xml",
        route_file="nets/big-intersection/routes.rou.xml",
        out_csv_name="outputs/big-intersection/test",
        use_gui=True,
        num_seconds=train_timeout,
        delta_time=sumo_tstep
    )
    eval_env = sumo_rl.parallel_env(
        net_file="nets/big-intersection/big-intersection.net.xml",
        route_file="nets/big-intersection/routes.rou.xml",
        out_csv_name="outputs/big-intersection/eval",
        use_gui=True,
        num_seconds=eval_timeout,
        delta_time=sumo_tstep
    )
    env = ss.pettingzoo_env_to_vec_env_v0(env)
    env = ss.concat_vec_envs_v0(
        env, num_envs, num_cpus=num_cpus, base_class="stable_baselines3"
    )
    env = VecMonitor(env)
    eval_env = ss.pettingzoo_env_to_vec_env_v0(eval_env)
    eval_env = ss.concat_vec_envs_v0(
        eval_env, 1, num_cpus=1, base_class="stable_baselines3"
    )
    eval_env = VecMonitor(eval_env)


    model = PPO(
        CustomActorCriticPolicy,
        env,
        verbose=3,
        gamma=0.99,
        n_steps=256,
        ent_coef=0.0905168,
        learning_rate=0.0001,
        vf_coef=0.042202,
        max_grad_norm=0.9,
        gae_lambda=0.9,
        n_epochs=5,
        clip_range=0.25,
        batch_size=256,
        tensorboard_log=save_path + "tensorboard",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path + "best_model",
        log_path=save_path + "eval",
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
        model.save(save_path + "final_model")
