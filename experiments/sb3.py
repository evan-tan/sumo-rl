import json
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import psutil
import sumo_rl
import supersuit as ss
import torch

# from custom.callbacks import EvalCallback
from custom.callbacks import TensorboardCallback
from custom.sb3_model import CustomActorCriticPolicy
from custom.utils import load_cfg, smooth_data
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecMonitor

if __name__ == "__main__":
    sumo_tstep = 7
    n_evaluations = 20
    num_cpus = 1  # int(psutil.cpu_count() - 1)
    # You can not use LIBSUMO if using more than one env
    n_envs = int(num_cpus - 4)
    # set this to the same as generator.py
    train_timesteps = int(1e5)
    eval_timesteps = int(1e3)
    n_agents = 1

    env = sumo_rl.parallel_env(
        net_file="nets/big-intersection/big-intersection.net.xml",
        route_file="nets/big-intersection/routes.rou.xml",
        out_csv_name="outputs/big-intersection/train",
        use_gui=True,
        num_seconds=train_timesteps,
        delta_time=sumo_tstep,
        yellow_time=2,
        min_green=5,
        max_green=100,
        max_depart_delay=0,
    )
    env = ss.pettingzoo_env_to_vec_env_v0(env)
    env = ss.concat_vec_envs_v0(
        env, n_envs, num_cpus=num_cpus, base_class="stable_baselines3"
    )
    env = VecMonitor(env)

    eval_env = sumo_rl.parallel_env(
        net_file="nets/big-intersection/big-intersection.net.xml",
        route_file="nets/big-intersection/routes.rou.xml",
        out_csv_name="outputs/big-intersection/eval",
        use_gui=True,
        num_seconds=eval_timesteps,
        delta_time=sumo_tstep,
        yellow_time=2,
        min_green=5,
        max_green=100,
        max_depart_delay=0,
    )
    eval_env = ss.pettingzoo_env_to_vec_env_v0(eval_env)
    eval_env = ss.concat_vec_envs_v0(
        eval_env, 1, num_cpus=1, base_class="stable_baselines3"
    )
    eval_env = VecMonitor(eval_env)
    # eval_freq = max(eval_freq // (n_envs * n_agents), 1)
    eval_freq = int(train_timesteps / n_evaluations)

    model = PPO(
        CustomActorCriticPolicy,
        env,
        verbose=0,
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
        tensorboard_log="./sb3_ppo/tensorboard/",
    )

    save_path = "./logs/"
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path + "best_model",
        log_path=save_path,
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=eval_freq, save_path=save_path + "checkpoints"
    )

    cb_chain = CallbackList([checkpoint_callback, eval_callback])
    model.learn(total_timesteps=train_timesteps, callback=cb_chain)

    model.learn(
        total_timesteps=train_timesteps
    )  # , callback=TensorboardCallback(0))# callback=eval_callback)
    # save a learned model
    save_path = "outputs/" + "MLPModel"
    model.save(save_path)

    train = False
    save_path = save_path + "final"
    if train:
        start_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        print(f"Start @ {start_time}")

        model.learn(total_timesteps=train_timesteps * 100, callback=eval_callback)
        end_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        print(f"End @ {end_time}")

        # save a learned model
        model.save(save_path)

    del model
    model_path = save_path + "best_model"
    model = PPO.load(model_path)

    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=10, callback=TensorboardCallback(0)
    )

    print(f"Reward Mean = {mean_reward:.3f}")
    print(f"Reward Std Dev = {std_reward:.3f}")

    # RUN EVALUATION ON SINGLE EPISODE
    n_episodes = 1
    obs = eval_env.reset()
    step_rewards = []
    n_sumo_timesteps = 5000 // sumo_tstep
    # actual iterations we want
    for i in range(n_sumo_timesteps):
        action, states = model.predict(obs)
        obs, rewards, dones, info = eval_env.step(action)
        step_rewards.append(rewards)
    episode_rewards = np.stack(step_rewards).squeeze()

    fig, ax = plt.subplots(figsize=(9, 9))
    y = episode_rewards
    x = np.linspace(0, y.size * sumo_tstep, y.size * sumo_tstep)
    ax.plot(x, y, alpha=0.5)
    ax.plot(x, smooth_data(y, 0.9))
    plt.show()
