"""This file just serves as an example on how to configure the zoo
using python scripts instead of yaml files."""

from eml_rl.Reward import RewardFunction
from f1tenth_gym.envs.track import Track
from eml_rl.f1tenth_transforms import F1TenthActionTransform, F1TenthObsTransform
# track = Track.from_track_name("Spielberg")
# raceline = track.raceline
# waypoints = np.stack([raceline.xs, raceline.ys, raceline.vxs], axis=1)
# rewarder = Rewarder(waypoints) "f1tenth_transforms.F1TenthActionTransform", 
hyperparams = {
    "f1tenth-v0": dict(
        env_wrapper=["eml_rl.f1tenth_transforms.F1TenthObsTransform", "eml_rl.f1tenth_transforms.F1TenthActionTransform"],
        # normalize=False,
        # n_envs=1,
        n_timesteps=200000.0,
        policy="MlpPolicy",


        # gamma=0.9,
        # learning_rate=0.003551345254057443,
        # batch_size=100,
        # buffer_size=10000,
        # tau=0.005,
        # train_freq=1,
        # 
        # noise_type=None,
        # noise_std =0.05673586673897195,
        # policy_kwargs={
        # "net_arch":[64,64],

        # },
        env_kwargs={
        "config": {
        # "reset_config": {"type": "rl_random_static"},
                "reward_function": RewardFunction,
                "map": "Spielberg",
                "num_agents": 1,
                "timestep": 0.01,
                "model": "st",
                "params": {"mu":1.0},
                "control_input": ["speed", "steering_angle"],
                "observation_config": {"type": "features",
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
                                    ]},
            },
        # "render_mode":"human",

        },
        eval_env_kwargs={
        "config": {
                "reward_function": RewardFunction,
                "map": "Spielberg",
                "num_agents": 1,
                "timestep": 0.01,
                "model": "st",
                "params": {"mu":1.0},
                "control_input": ["speed", "steering_angle"],
                "observation_config": {"type": "features",
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
                                    ]},
            },
        }
    )
}
