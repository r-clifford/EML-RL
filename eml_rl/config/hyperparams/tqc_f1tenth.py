from eml_rl.utils import basic_config

policy_kwargs = dict(net_arch=[1600, 1200])
use_sde = True


train_conf = basic_config()
eval_conf = basic_config()
eval_conf["reset_config"] = {"type": "cl_grid_static"}
hyperparams = {
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
            {"gymnasium.wrappers.FrameStack": {
                "num_stack": train_conf["frame_stack"]}},
        ],
        callback=["eml_rl.f1tenth_transforms.F1TenthTensorboardCallback"],
        # normalize=False,
        # n_envs=1,
        n_timesteps=25000.0,
        policy="MlpPolicy",
        policy_kwargs=policy_kwargs,
        use_sde=use_sde,
        env_kwargs={"config": train_conf["config"]},
        eval_env_kwargs={
            "config": eval_conf["config"],
        },
    )
}
