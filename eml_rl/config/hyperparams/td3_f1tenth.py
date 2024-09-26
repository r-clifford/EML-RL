from eml_rl.config.f1tenth_config import get_default_hyperparams

policy_kwargs = dict(net_arch=[1600, 1200, 800])


params = dict(
#     gamma=0.999,
    learning_rate=0.000014637059120891274,
#     batch_size=128,
#     buffer_size=1000000,
#     learning_starts=0,
    train_freq=2,
    policy_kwargs=policy_kwargs,
)
hyperparams = get_default_hyperparams()

hyperparams["f1tenth-v0"].update(params)
