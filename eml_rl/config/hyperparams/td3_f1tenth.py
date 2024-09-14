"""This file just serves as an example on how to configure the zoo
using python scripts instead of yaml files."""

from eml_rl.Reward import ProgressReward
from eml_rl.f1tenth_transforms import F1TenthActionTransform, F1TenthObsTransform

hyperparams = {
    "f1tenth-v0": dict(
        env_wrapper=[
            "eml_rl.f1tenth_transforms.F1TenthObsTransform",
            "eml_rl.f1tenth_transforms.F1TenthActionTransform",
            {"gymnasium.wrappers.FrameStack": {"num_stack": 5}},
        ],
        # normalize=False,
        # n_envs=1,
        n_timesteps=200000.0,
        policy="MlpPolicy",
        env_kwargs={
            "config": {
                "reset_config": {"type": "rl_random_static"},
                "reward_function": ProgressReward(),
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
        },
        eval_env_kwargs={
            "config": {
                "reward_function": ProgressReward(),
                "map": "Catalunya",
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
        },
    )
}
