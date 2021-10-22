import json

import numpy as np
import sumo_rl


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
    net = paths.get('net')
    route = paths.get('route')
    out_csv = paths.get('output_csv')

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
