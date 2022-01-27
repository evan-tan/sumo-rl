import json
import re
from typing import Callable, List

import numpy as np
import pettingzoo
import sumo_rl
import supersuit as ss
import yaml
from attrdict import AttrDict
from stable_baselines3.common.vec_env import VecMonitor


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


def load_cfg(_file: str) -> dict:
    """Load config.json
    Args:
        file (str): file path to YAML config
    Returns:
        dict: config stuff
    """

    cfg = None
    with open(_file) as f:
        if ".json" in _file:
            cfg = json.load(f)
        elif ".yaml" in _file or ".yml" in _file:
            _loader = yaml.SafeLoader
            _loader.add_implicit_resolver(
                u"tag:yaml.org,2002:float",
                re.compile(
                    u"""^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$""",
                    re.X,
                ),
                list(u"-+0123456789."),
            )
            _tmp = yaml.load(f, Loader=_loader)
            cfg = AttrDict(_tmp)
    return cfg


def unsqueeze(arr: np.ndarray, dim: int) -> np.ndarray:
    """Wrapper function for torch.unsqueeze() functionality in NumPy"""
    return np.expand_dims(arr, axis=dim)


def env_creator(cfg: yaml, mode: str) -> pettingzoo.AECEnv:
    """For PettingZoo environments

    Args:
        cfg    (yaml): json file containing all relevant keys
        mode    (str): str representing "train" or "eval"
    Returns:
        pettingzoo.AECEnv: PettingZoo parallel environment
    """

    # time unit conversion between simulators
    YELLOW_TIME = 2
    GREEN_TIME = 5
    assert cfg.env_tstep == (YELLOW_TIME + GREEN_TIME)

    net = cfg.paths.net_file
    route = cfg.paths.route_file
    out_csv = cfg.paths.output_path

    assert mode in cfg.keys()
    # dependent on train/eval mode
    subconfig = cfg.get(mode)
    subconfig = AttrDict(subconfig)
    # actual number of seconds in SUMO before time out
    timeout = int(subconfig.env_timeout)

    if cfg.env_parallel:
        env = sumo_rl.parallel_env(
            net_file=net,
            route_file=route,
            out_csv_name=out_csv,
            use_gui=True,
            num_seconds=timeout,
            delta_time=cfg.env_tstep,
            yellow_time=YELLOW_TIME,
            min_green=GREEN_TIME,
            max_green=50,
        )
    else:
        env = sumo_rl.env(
            net_file=net,
            route_file=route,
            out_csv_name=out_csv,
            use_gui=True,
            num_seconds=timeout,
            delta_time=cfg.env_tstep,
            yellow_time=YELLOW_TIME,
            min_green=GREEN_TIME,
            max_green=50,
        )

    # OVERRIDE num_envs and num_cpus for eval mode
    num_envs = 1 if mode == "eval" else cfg.num_envs
    num_cpus = 1 if mode == "eval" else cfg.num_cpus

    env = ss.pettingzoo_env_to_vec_env_v0(env)
    env = ss.concat_vec_envs_v0(
        env, num_envs, num_cpus=num_cpus, base_class="stable_baselines3"
    )
    env = VecMonitor(env)
    return env


if __name__ == "__main__":
    cfg = load_cfg("../config.yml")
    print(cfg)
    print(cfg.train.env_timeout)
    print(cfg.paths.net_file)
    print(cfg.train)
    print("train" in cfg.keys())
    print(cfg.get("train"))
    test_env = env_creator(cfg, "train")
