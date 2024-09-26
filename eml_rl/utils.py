import gymnasium as gym
import numpy as np


from eml_rl.f1tenth_transforms import (
    F1TenthObsTransform,
    F1TenthActionTransform,
    FrameSkip,
)
from eml_rl.reward import ProgressReward
from gymnasium.wrappers import FrameStack
from stable_baselines3.common.utils import set_random_seed


def basic_config():
    conf = {
        "config": {
            "params_randomizer": randomize_sim_params(0.1),
            "params": {"mu": 0.3},
            "reset_config": {"type": "shuf_random_static"},
            "reward_class": ProgressReward(),
            "map": "Oschersleben",
            "num_agents": 1,
            "timestep": 0.01,
            "model": "st",
            "control_input": ["speed", "steering_angle"],
            "observation_config": {
                "type": "features",
                "features": [
                    "pose_x",
                    "pose_y",
                    "scan",
                    "pose_theta",
                    "linear_vel_x",
                    "ang_vel_z",
                    "collision",
                    "lap_time",
                    "lap_count",
                ],
            },
        },
        "frame_stack": 10,
        "frame_skip": 3,
        "lidar_beams": 80,
        "vmax": 8.0,
        "vmin": 1.0,
    }
    return conf.copy()


def randomize_sim_params(percent: float):
    """Randomize params where value is normally distributed with sigma = percent * orig_value

    Args:
        percent: Percentage of original value to use as sigma

    Returns:
        function: Function to randomize parameters
    """

    def f(params):
        params = params.copy()
        for key in params:
            if key not in ("width", "length"):
                val = params[key]
                sigma = percent * val
                val = val + np.random.normal(0, abs(sigma))
                params[key] = val
        return params

    return f


def make_env(env_id: str, rank: int, seed: int = 0):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param seed: the initial seed for RNG
    :param rank: index of the subprocess
    """

    def _init():
        conf = basic_config()
        env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config=conf["config"],
            render_mode="rgb_array",
        )

        env = F1TenthObsTransform(env, beam_count=conf["lidar_beams"])
        env = FrameStack(env, conf["frame_stack"])
        env = F1TenthActionTransform(env, vmax=conf["vmax"], vmin=conf["vmin"])
        env = FrameSkip(env, (2, 4))
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init


def default_sim_params():
    """Get dictionary of default simulation parameters

    Returns:
        dict: Dictionary of default simulation parameters
    """
    return {
        "mu": 1.0489,
        "C_Sf": 4.718,
        "C_Sr": 5.4562,
        "lf": 0.15875,
        "lr": 0.17145,
        "h": 0.074,
        "m": 3.74,
        "I": 0.04712,
        "s_min": -0.4189,
        "s_max": 0.4189,
        "sv_min": -3.2,
        "sv_max": 3.2,
        "v_switch": 7.319,
        "a_max": 9.51,
        "v_min": -5.0,
        "v_max": 20.0,
        "width": 0.31,
        "length": 0.58,
    }
