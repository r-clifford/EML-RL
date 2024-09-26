from eml_rl.utils import basic_config


def get_default_hyperparams():
    train_conf = basic_config().copy()
    eval_conf = basic_config().copy()
    eval_conf["reset_config"] = {"type": "cl_grid_static"}
    return {
        "f1tenth-v0": dict(
            env_wrapper=[
                {
                    "eml_rl.f1tenth_transforms.F1TenthObsTransform": {
                        "beam_count": train_conf["lidar_beams"],
                    }
                },
                {
                    "eml_rl.f1tenth_transforms.F1TenthActionTransform": {
                        "vmax": train_conf["vmax"],
                        "vmin": train_conf["vmin"],
                    }
                },
                {
                    "gymnasium.wrappers.FrameStack": {
                        "num_stack": train_conf["frame_stack"]
                    }
                },
                {
                    "eml_rl.f1tenth_transforms.FrameSkip": {
                        "skip": train_conf["frame_skip"]
                    }
                },
            ],
            callback=["eml_rl.f1tenth_transforms.F1TenthTensorboardCallback"],
            # normalize=False,
            # n_envs=1,
            n_timesteps=25000.0,
            policy="MlpPolicy",
            policy_kwargs=None,
            env_kwargs={"config": train_conf["config"]},
            eval_env_kwargs={
                "config": eval_conf["config"],
            },
        )
    }
