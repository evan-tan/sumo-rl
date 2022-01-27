import json
from typing import Callable, List

import numpy as np
import sumo_rl
from stable_baselines3 import PPO


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


def smooth_data(scalars: List[float], weight: float) -> List[float]:
    """Tensorboard smoothing function to smooth noisy training data

    :param scalars: data points to smooth
    :type scalars: List[float]
    :param weight: Exponential Moving Average weight in 0-1
    :type weight: float
    :return: smoothed data points
    :rtype: List[float]
    """
    assert weight >= 0 and weight <= 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val

    return smoothed


def load_cfg(json_file) -> np.ndarray:
    with open(json_file, "r") as f:
        return json.load(f)


def unsqueeze(arr: np.ndarray, dim: int) -> np.ndarray:
    """Wrapper function for torch.unsqueeze() functionality in NumPy"""
    return np.expand_dims(arr, axis=dim)


def env_creator(num_timesteps, paths, parallel=False):
    """For PettingZoo"""
    env_timestep = 5
    assert len(paths.keys()) != 0
    net = paths.get("net")
    route = paths.get("route")
    out_csv = paths.get("output_csv")

    if parallel:
        env = sumo_rl.parallel_env(
            net_file=net,
            route_file=route,
            out_csv_name=out_csv,
            use_gui=True,
            num_seconds=int(num_timesteps),
            delta_time=env_timestep,
            yellow_time=2,
            min_green=5,
            max_green=50,
        )
    else:
        env = sumo_rl.env(
            net_file=net,
            route_file=route,
            out_csv_name=out_csv,
            use_gui=True,
            num_seconds=int(num_timesteps),
            delta_time=env_timestep,
            yellow_time=2,
            min_green=5,
            max_green=50,
        )

    return env
