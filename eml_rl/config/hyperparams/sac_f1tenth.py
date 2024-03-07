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
        # normalize=True,
        normalize=False,
        n_envs=1,
        n_timesteps=20000.0,
        policy="MlpPolicy",
        batch_size=128,
        gamma=0.95,
        learning_rate=0.000474369265241482730,
        ent_coef=0.00429,
        buffer_size=10000,
        learning_starts=0,
        train_freq=8,
        tau=0.01,
        # max_grad_norm=5,
        # vf_coef=0.19,
        use_sde=True,
        policy_kwargs={
                "net_arch":[400,300],
"log_std_init": -2.4189717788859246
                },

        env_kwargs={
        "config": {
        "reset_config": {"type": "rl_random_static"},
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
