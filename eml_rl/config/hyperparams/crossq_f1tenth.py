"""
Best trial:
Value:  94.34377959999999
Params:
    gamma: 0.995
    learning_rate: 7.347956920516056e-05
    batch_size: 128
    buffer_size: 10000
    learning_starts: 1000
    train_freq: 64
    tau: 0.02
    log_std_init: -2.3116786232284
    net_arch: f10_small
    batch_norm_momentum: 0.08851596904271067
    batch_norm_eps: 0.00020296247684742797
Writing report to precommit-test/zoo-test/crossq-1726966452/crossq/report_f1tenth-v0_500-trials-50000-tpe-median_1726969448
"""
from eml_rl.utils import basic_config


policy_kwargs = dict(net_arch=[1600, 1200], use_expln=True)
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
            {"gymnasium.wrappers.FrameStack": {"num_stack": train_conf["frame_stack"]}},
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
