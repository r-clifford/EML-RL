import gymnasium as gym
import numpy as np
import gymnasium as gym
from Reward import RewardFunction


class F1TenthActionTransform(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        low = np.array([[-0.4189, 0.0]]).astype(np.float32)
        high = np.array([[0.4189, 10.0]]).astype(np.float32)
        env.action_space = gym.spaces.Box(low=low, high=high, shape=(1,2), dtype=np.float32)

    def action(self, action):
        return action
    
class F1TenthObsTransform(gym.ObservationWrapper):
    def __init__(self, env):
        super(F1TenthObsTransform, self).__init__(env)
        self.last = np.zeros((40, ))
        self.observation_space = gym.spaces.Box(low=0.0, high=30.0, shape=(84, ))
        self.scale = 10
        # print("OBS: ", self.observation_space.shape)

    def observation(self, observation):
        scan: gym.spaces.Box = observation["agent_0"]["scan"]
        # scan = np.reshape(scan,(20, 1080//20))
        # scan = np.mean(scan, axis=1)
        # scan = np.ndarray.flatten(scan)
        scan = scan/self.scale
        scan = np.clip(scan, 0, 1)
        indices = np.linspace(0, 1079, 40, dtype=int)
        scan = scan[indices]
        tmp = scan
        scan = np.concatenate((self.last,scan))
        self.last = tmp
        state = np.array([observation["agent_0"]["linear_vel_x"], observation["agent_0"]["ang_vel_z"],0,0])
        state = np.concatenate((state, scan))
        # print(state)
        # print(f"arr: {np.array(scan)}")
        return state

env = gym.make(
    "f1tenth_gym:f1tenth-v0",
    config={
        "reset_config": {"type": "rl_random_static"},
        "reward_function": RewardFunction,
        "map": "Spielberg",
        "num_agents": 1,
        "timestep": 0.01,
        "model": "st",
        "params": {"mu":1.0},
        "control_input": ["speed", "steering_angle"],
        "observation_config": {"type": "features",
                               "features": [
                                   "pose_x",
                                   "pose_y",
                                   "scan",
                                   "pose_theta",
                                   "linear_vel_x",
                                   "ang_vel_z",
                                   "collision",
                                   "lap_time",
                                   "lap_count",
                               ]},
    },
    render_mode="rgb_array",
)


env = F1TenthObsTransform(env)

low = np.array([[-0.4189, 0.0]]).astype(np.float32)
high = np.array([[0.4189, 10.0]]).astype(np.float32)
env = F1TenthActionTransform(env)
from stable_baselines3 import TD3, SAC, PPO
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.02 * np.ones(n_actions))


model = SAC.load("/".join(__file__.split("/")[:-1]) + "/best_model.zip", env)
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



