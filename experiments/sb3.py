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
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecMonitor

# NOTE: Don't forget to execute this script from 1 directory above experiments/

if __name__ == "__main__":
    # TODO: store stuff in json maybe
    cfg = load_cfg("./config.json")

    sumo_tstep = 7
    n_evaluations = 20
    num_cpus = int(psutil.cpu_count() - 1)
    # You can not use LIBSUMO if using more than one env
    n_envs = 3#int(num_cpus - 4)
    train_timesteps = int(1e5)
    eval_timesteps = int(1e3)
    n_agents = 1

    env = sumo_rl.parallel_env(
        net_file="nets/big-intersection/big-intersection.net.xml",
        route_file="nets/big-intersection/routes.rou.xml",
        out_csv_name="outputs/big-intersection/train",
        # net_file='nets/2way-single-intersection/single-intersection.net.xml',
        # route_file='nets/2way-single-intersection/single-intersection-gen.rou.xml',
        # out_csv_name='outputs/2way-single-intersection/a2c',
        use_gui=True,
        num_seconds=train_timesteps,
        delta_time=sumo_tstep, 
        yellow_time=2,
        min_green=5,
        max_green=100,
        max_depart_delay=0
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
        # net_file='nets/2way-single-intersection/single-intersection.net.xml',
        # route_file='nets/2way-single-intersection/single-intersection-gen.rou.xml',
        # out_csv_name='outputs/2way-single-intersection/a2c',
        use_gui=True,
        num_seconds=eval_timesteps,
        delta_time=sumo_tstep, 
        yellow_time=2,
        min_green=5,
        max_green=100,
        max_depart_delay=0
    )
    eval_env = ss.pettingzoo_env_to_vec_env_v0(eval_env)
    eval_env = ss.concat_vec_envs_v0(
        eval_env, 1, num_cpus=1, base_class="stable_baselines3"
    )
    eval_env = VecMonitor(eval_env)

    eval_freq = int(train_timesteps / n_evaluations)
    eval_freq = max(eval_freq // (n_envs*n_agents), 1)

    # TODO: replace with custom policy_kwargs
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

    eval_save_path = "./logs/"
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=eval_save_path,
        log_path=eval_save_path,
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
    )
    
    
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback
    
    from torch.utils.tensorboard import SummaryWriter

    class TensorboardCallback(BaseCallback):
        """
        Custom callback for plotting additional values in tensorboard.
        """

        def __init__(self, verbose=0):
            super(TensorboardCallback, self).__init__(verbose)
            self.raw_envs = eval_env.unwrapped.vec_envs
            self.envs = []
            for i in range(len(self.raw_envs)):
                self.envs.append(self.raw_envs[i].par_env.aec_env.env.env.env)
                
            self.tb_writer = SummaryWriter(str("./Sumo"))
            #self.metric_names = list(self.envs[0].metrics[-1].keys())
            self.iter = 0
            
            
        def find_and_record(self, name):
            wait_times = [env.metrics[-1][name] for env in self.envs]
            wait_time_names = [name + '/' + str(i) for i in range(len(self.envs))]
            for i, name in enumerate(wait_time_names):
                item = wait_times[i]
                if type(item) == list:
                    item = sum(item)
                self.logger.record(name, item)
            

        def _on_step(self) -> bool:

            #Log scalar value (here a random variable)

            return True
        
        def __call__(self, a, b):
            metrics = a['infos'][0]
            
            #if len(metrics) != 0:
            #    for item in list(metrics.keys())[2:]:
            #        self.find_and_record(item)
            #else:
            #    self.iter += 1

            #self.logger.dump(self.num_timesteps + self.iter*eval_timesteps)
            
            #worker_number = self.env.label
            run_number = 1
            
            max_sumo_timestep = eval_timesteps# self.env.sim_max_time
            current_timestep = self.num_timesteps
            
            total_sumo_timestep = current_timestep + run_number*max_sumo_timestep
            #last_info = self.env.metrics[-1]
            
            log_metric_names = list(metrics.keys())[1:]

            for key, val in metrics.items():
                #key += "/" + str(worker_number)
                if type(val) is list or type(val) is np.ndarray:
                    val = np.mean(val)
                self.tb_writer.add_scalar(key, val, current_timestep)
            #print("added")
            
            print("tru")
        
        
    
    
    model.learn(total_timesteps=train_timesteps)#, callback=TensorboardCallback(0))# callback=eval_callback)
    # save a learned model
    save_path = "outputs/" + cfg.get("model_name")
    model.save(save_path)

    train = False
    save_path = eval_save_path + "final"
    if train:
        start_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        print(f"Start @ {start_time}")

        model.learn(total_timesteps=train_timesteps * 100, callback=eval_callback)
        end_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        print(f"End @ {end_time}")

        # save a learned model
        model.save(save_path)

    del model
    model_path = eval_save_path + "best_model"
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

    """ render_env = sumo_rl.env(net_file='nets/4x4-Lucas/4x4.net.xml',
                        route_file='nets/4x4-Lucas/4x4c1c2c1c2.rou.xml',
                        out_csv_name='outputs/4x4grid/test',
                        use_gui=False,
                        num_seconds=80000)

    render_env = render_env.parallel_env()
    render_env = ss.color_reduction_v0(render_env, mode='B')
    render_env = ss.resize_v0(render_env, x_size=84, y_size=84)
    render_env = ss.frame_stack_v1(render_env, 3)

    obs_list = []
    i = 0
    render_env.reset()


    while True:
        for agent in render_env.agent_iter():
            observation, _, done, _ = render_env.last()
            action = model.predict(observation, deterministic=True)[0] if not done else None

            render_env.step(action)
            i += 1
            if i % (len(render_env.possible_agents)) == 0:
                obs_list.append(np.transpose(render_env.render(mode='rgb_array'), axes=(1, 0, 2)))
        render_env.close()
        break

    print('Writing gif')
    write_gif(obs_list, 'kaz.gif', fps=15) """
