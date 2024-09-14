import gymnasium as gym
import numpy as np
from gymnasium.wrappers import FrameStack
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from Reward import ProgressReward
from f1tenth_transforms import *

SEED = 101
# TODO: Implement Rewarder base class
# Decouple reward function from the environment
# Init rewarder with env params
# Only pass observation to reward function
# Example: Implement follow the gap, wall follow, or disparity
# through writing reward functions
# Consider punishing changing sign of steering angle
#
# Organize trained models in directory


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
                "reward_class": ProgressReward(),
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
    # env = SubprocVecEnv([make_env("f1tenth_gym:f1tenth-v0", i) for i in range(1)])
    # env = gym.make(
    #     "f1tenth_gym:f1tenth-v0",
    #     config={
    #         "reset_config": {"type": "rl_random_static"},
    #         "reward_function": RewardFunction,
    #         "map": "Spielberg",
    #         "num_agents": 1,
    #         "timestep": 0.03,
    #         "model": "st",
    #         "control_input": ["speed", "steering_angle"],
    #         "observation_config": {
    #             "type": "features",
    #             "features": [
    #                 "pose_x",
    #                 "pose_y",
    #                 "scan",
    #                 "pose_theta",
    #                 "linear_vel_x",
    #                 "ang_vel_z",
    #                 "collision",
    #                 "lap_time",
    #                 "lap_count",
    #             ],
    #         },
    #     },
    #     render_mode="rgb_array",
    # )

    env = make_env("train", 0, SEED)()
    # env = DummyVecEnv([make_env("f1tenth_gym:f1tenth-v0", i) for i in range(2)])

    checkpoint_callback = CheckpointCallback(save_freq=50000, save_path="./logs/")
    # Separate evaluation env
    # eval_env = make_env("eval", 1, 1)()
    # eval_env = SubprocVecEnv([make_env("eval", 10+i) for i in range(2)])
    # eval_callback = EvalCallback(
    #     eval_env,
    #     best_model_save_path="./logs/best_model",
    #     log_path="./logs/results",
    #     eval_freq=25000,
    # )
    # Create the callback list
    callback = CallbackList(
        [
            checkpoint_callback,
            # eval_callback
        ]
    )
    # low = np.array([[-0.4189, 0.0]]).astype(np.float32)
    # high = np.array([[0.4189, 10.0]]).astype(np.float32)
    from stable_baselines3 import TD3, SAC, PPO
    from stable_baselines3.common.noise import (
        NormalActionNoise,
        OrnsteinUhlenbeckActionNoise,
    )

    n_actions = env.action_space.shape[-1]
    # action_noise = NormalActionNoise(
    #     mean=np.zeros(n_actions), sigma=0.05 * np.ones(n_actions)
    # )
    # policy_kwargs = dict(net_arch=[512, 512])
    policy_kwargs = dict(net_arch=[1600, 1200])
    # model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="logs", policy_kwargs=policy_kwargs)
    model = TD3(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="logs",
        policy_kwargs=policy_kwargs,
        # action_noise=action_noise,
    )
    try:
        model.learn(
            total_timesteps=int(10 * 6 * 1e4), progress_bar=True, 
            callback=callback
        )
    finally:
        model.save("td3_f1tenth")
    #
    # model = SAC.load("/".join(__file__.split("/")[:-1]) + "/best_model.zip", env)
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
