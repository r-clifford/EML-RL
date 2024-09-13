import gymnasium as gym
import numpy as np
from gymnasium.wrappers import FrameStack
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from Reward import RewardFunction
from f1tenth_transforms import *
import sys

def make_env(env_id: str, rank: int, seed: int = 0):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the initial seed for RNG
    :param rank: index of the subprocess
    """

    def _init():
        # env = gym.make(env_id, render_mode="human")
        env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config={
                "reset_config": {"type": "rl_random_static"},
                "reward_function": RewardFunction,
                "map": "Catalunya",
                "num_agents": 1,
                "timestep": 0.03,
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
            render_mode="rgb_array",
        )

        env = F1TenthObsTransform(env)
        env = FrameStack(env, 5)
        env = F1TenthActionTransform(env)
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init


if __name__ == "__main__":
    env = make_env("train", 0, 0)()

    # low = np.array([[-0.4189, 0.0]]).astype(np.float32)
    # high = np.array([[0.4189, 10.0]]).astype(np.float32)
    from stable_baselines3 import TD3, SAC, PPO
    from stable_baselines3.common.noise import (
        NormalActionNoise,
        OrnsteinUhlenbeckActionNoise,
    )

    n_actions = env.action_space.shape[-1]
    model = TD3.load(sys.argv[1], env)
    vec_env = model.get_env()
    #
    i = 0.0
    obs = vec_env.reset()
    while True:
        try:
            action, _states = model.predict(obs)
            print(action)
            obs, rewards, dones, info = vec_env.step(action)
            vec_env.render("human")
            i += 0.01
            if i > 60:
                i = 0.0
                env.reset()
        except KeyboardInterrupt:
            break
