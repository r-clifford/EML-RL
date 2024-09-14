import gymnasium as gym
import numpy as np

class F1TenthActionTransform(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        low = np.array([[-0.4189, 1.0]]).astype(np.float32)
        high = np.array([[0.4189, 8.0]]).astype(np.float32)
        env.action_space = gym.spaces.Box(low=low, high=high, shape=(1,2), dtype=np.float32)

    def action(self, action):
        # print("1")
        # return np.array([])
        return action
    
class F1TenthObsTransform(gym.ObservationWrapper):
    def __init__(self, env):
        super(F1TenthObsTransform, self).__init__(env)
        self.last = np.zeros((40, ))
        # self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(80, ))
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(40, ))
        self.scale = 30
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
        # tmp = scan
        # scan = np.concatenate((self.last,scan))
        # self.last = tmp
        # state = np.array([observation["agent_0"]["linear_vel_x"], observation["agent_0"]["ang_vel_z"],0,0])
        # state = np.concatenate((state, scan))
        # print(state)
        # print(f"arr: {np.array(scan)}")
        return scan
