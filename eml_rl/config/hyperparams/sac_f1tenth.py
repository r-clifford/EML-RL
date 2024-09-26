from eml_rl.config.f1tenth_config import get_default_hyperparams

policy_kwargs = dict(net_arch=[1600, 1200, 800])
use_sde = True
params = dict(
    policy_kwargs=policy_kwargs,
    use_sde=use_sde,
)

hyperparams = get_default_hyperparams()
hyperparams["f1tenth-v0"].update(params)
