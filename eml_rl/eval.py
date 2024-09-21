import sys
import numpy as np
from eml_rl.utils import make_env

if __name__ == "__main__":
    env = make_env("eval", 0, 0)()

    from stable_baselines3 import TD3, SAC, PPO
    from sb3_contrib import TQC, RecurrentPPO
    from stable_baselines3.common.noise import (
        NormalActionNoise,
        OrnsteinUhlenbeckActionNoise,
    )

    n_actions = env.action_space.shape[-1]
    if len(sys.argv) < 2:
        print("Usage: python eval.py <algo> <model_path>")
        exit(1)
    algo = sys.argv[1]
    if algo == "td3":
        model = TD3.load(sys.argv[2], env)
    elif algo == "sac":
        model = SAC.load(sys.argv[2], env)
    elif algo == "ppo":
        model = PPO.load(sys.argv[2], env)
    elif algo == "tqc":
        model = TQC.load(sys.argv[2], env)
    elif algo == "rppo":
        model = RecurrentPPO.load(sys.argv[2], env)
    else:
        print("Usage: python eval.py <algo> <model_path>")
        exit(1)
    vec_env = model.get_env()
    #
    i = 0.0
    lstm_state = None
    episodes_starts = np.ones((vec_env.num_envs,), dtype=bool)
    obs = vec_env.reset()
    while True:
        try:
            if algo in ["rppo"]:
                action, lstm_state = model.predict(obs, state=lstm_state, mask=episodes_starts)
            action, _ = model.predict(obs)
            obs, rewards, dones, info = vec_env.step(action)
            episodes_starts = dones

            vec_env.render("human")
            i += 0.01
            if i > 60:
                i = 0.0
                env.reset()
        except KeyboardInterrupt:
            break
