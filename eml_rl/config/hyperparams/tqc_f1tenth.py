"""
Number of finished trials:  25
Best trial:
Value:  272.98810315
Params:
    gamma: 0.95
    learning_rate: 0.0002896379388101884
    batch_size: 128
    buffer_size: 10000
    learning_starts: 1000
    train_freq: 512
    tau: 0.001
    log_std_init: -0.3456575394756052
    net_arch: f10_small
    n_quantiles: 27
    top_quantiles_to_drop_per_net: 23
Writing report to precommit-test/zoo-test/tqc-1726952990/tqc/report_f1tenth-v0_500-trials-50000-tpe-median_1726964734
"""
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
