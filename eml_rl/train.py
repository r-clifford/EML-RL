import sys
import stable_baselines3
import sb3_contrib
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from utils import make_env

SEED = 101

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train.py <algo>")
        exit(1)
    algo = sys.argv[1]
    env = make_env("train", 0, SEED)()
    # env = DummyVecEnv([make_env("f1tenth_gym:f1tenth-v0", i, np.random.randint(0,1000)) for i in range(2)])
    env = Monitor(env, info_keywords=("progress", "laptimes"))

    checkpoint_callback = CheckpointCallback(save_freq=50000, save_path=f"./logs/{algo}")
    # Separate evaluation env
    eval_env = make_env("eval", 1, 1)()
    eval_env = Monitor(
        eval_env, info_keywords=("progress", "laptimes"), filename="out.log"
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./logs/{algo}/best_model",
        log_path=f"./logs/{algo}/results",
        eval_freq=25000,
        deterministic=True,
        render=False,
        verbose=1,
        n_eval_episodes=10,
    )
    # Create the callback list
    callback = CallbackList(
        [
            checkpoint_callback,
            eval_callback,
        ]
    )
    from stable_baselines3 import TD3, SAC, PPO
    from stable_baselines3.common.noise import (
        NormalActionNoise,
        OrnsteinUhlenbeckActionNoise,
    )

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions), sigma=0.01 * np.ones(n_actions)
    )
    policy_kwargs = dict(net_arch=[1600, 1200])

    if algo == "sac":
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log="logs/sac",
            policy_kwargs=policy_kwargs,
            # train_freq=16,
            use_sde=True,
        )
    elif algo == "td3":
        model = TD3(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log="logs/td3",
            policy_kwargs=policy_kwargs,
            # batch_size=64,
            # buffer_size=1000000,
            # gamma=0.98,
            # learning_rate=1.8729356621045733e-05,
            # tau=0.05,
            # train_freq=16,
            # action_noise=action_noise,
        )
    elif algo == "rppo":
        model = sb3_contrib.RecurrentPPO(
            "MlpLstmPolicy",
            env,
            verbose=1,
            policy_kwargs=policy_kwargs,
            tensorboard_log="logs/rppo",
            use_sde=True,
        )
    elif algo == "ppo":
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            policy_kwargs=policy_kwargs,
            tensorboard_log="logs/ppo",
            use_sde=True,
        )
    elif algo == "tqc":
        model = sb3_contrib.TQC(
            "MlpPolicy",
            env,
            verbose=1,
            policy_kwargs=policy_kwargs,
            tensorboard_log="logs/tqc",
            use_sde=True,
        )
    else:
        print("Usage: python train.py <algo>")
        exit(1)
    try:
        model.learn(total_timesteps=int(1e6), progress_bar=True, callback=callback)
    finally:
        model.save(f"{algo}_f1tenth")
    vec_env = model.get_env()
    i = 0.0
    obs = vec_env.reset()
    while True:
        try:
            action, _states = model.predict(obs)
            print(action)
            obs, rewards, dones, info = vec_env.step(action)
            vec_env.render("human")
            i += 0.01
            if i > 60:
                i = 0.0
                env.reset()
        except KeyboardInterrupt:
            break
