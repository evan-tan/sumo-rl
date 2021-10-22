import json

import numpy as np
import sumo_rl


def load_cfg(json_file) -> np.ndarray:
    with open(json_file, "r") as f:
        return json.load(f)


def unsqueeze(arr: np.ndarray, dim: int) -> np.ndarray:
    """Wrapper function for torch.unsqueeze() functionality in NumPy"""
    return np.expand_dims(arr, axis=dim)


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
            delta_time=env_timestep,
            yellow_time=2,
            min_green=5,
            max_green=50,
        )
    else:
        env = sumo_rl.env(
            net_file="nets/big-intersection/big-intersection.net.xml",
            route_file="nets/big-intersection/routes.rou.xml",
            out_csv_name="outputs/big-intersection/test",
            use_gui=True,
            num_seconds=int(num_timesteps),
            delta_time=env_timestep,
            yellow_time=2,
            min_green=5,
            max_green=50,
        )

    return env
