"""
603.9613151499999
use_expln False

gamma 0.999

learning_rate 0.00014637059120891274

batch_size 512

buffer_size 1000000

learning_starts 0

train_freq 128

tau 0.005

log_std_init -3.6957093986510907

net_arch f10_medium

batch_norm_momentum 0.03218615615174704

batch_norm_eps 0.0001175151013153794

Value:  1256.4336043
Params:
    use_expln: False
    gamma: 0.98
    learning_rate: 0.00038753475519940516
    batch_size: 2048
    buffer_size: 1000000
    learning_starts: 0
    train_freq: 256
    tau: 0.001
    log_std_init: -3.8043073463910484
    net_arch: f10_small
    batch_norm_momentum: 0.05063481368886259
    batch_norm_eps: 0.00035669247850631604
Writing report to logs/crossq-1727219528/crossq/report_f1tenth-v0_500-trials-25000-tpe-median_1727238274
"""

from eml_rl.config.f1tenth_config import get_default_hyperparams


policy_kwargs = dict(
    net_arch=[1600, 1200, 800],
    use_expln=False,
    batch_norm_momentum=0.03218615615174704,
    batch_norm_eps=0.0001175151013153794,
    log_std_init=-3.6957093986510907,
)

use_sde = True

params = dict(
    #     gamma=0.999,
    learning_rate=0.000014637059120891274,
    #     batch_size=128,
    #     buffer_size=1000000,
    #     learning_starts=0,
    train_freq=2,
    policy_kwargs=policy_kwargs,
    use_sde=use_sde,
)

hyperparams = get_default_hyperparams()
hyperparams["f1tenth-v0"].update(params)
