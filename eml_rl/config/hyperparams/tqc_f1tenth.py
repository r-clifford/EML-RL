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
Number of finished trials:  198
Best trial:
Value:  1604.9653309999999
Params:
    use_expln: False
    gamma: 0.999
    learning_rate: 7.360450485873012e-05
    batch_size: 1024
    buffer_size: 10000
    learning_starts: 0
    train_freq: 512
    tau: 0.01
    log_std_init: -0.3044998886067871
    net_arch: f10_verylarge
    n_quantiles: 46
    top_quantiles_to_drop_per_net: 0
Writing report to logs/tqc-1727135504/tqc/report_f1tenth-v0_500-trials-25000-tpe-median_1727214473
"""
from eml_rl.config.f1tenth_config import get_default_hyperparams

use_sde = True
policy_kwargs = dict(
    net_arch=[3200, 2400, 1200],
    use_expln=False,
    use_sde=use_sde,
    # log_std_init=-0.3044998886067871,
    # n_quantiles=46,

)

use_sde = True

params = dict(
    # gamma=0.999,
    # learning_rate=7.360450485873012e-05,
    # batch_size=1024,
    # buffer_size=10000,
    # learning_starts=0,
    # train_freq=16,
    # top_quantiles_to_drop_per_net=0,
    # tau=0.01,
    policy_kwargs=policy_kwargs,
)

hyperparams = get_default_hyperparams()
hyperparams["f1tenth-v0"].update(params)
