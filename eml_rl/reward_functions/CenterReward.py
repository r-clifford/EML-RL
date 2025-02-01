from eml_rl.reward import EnvironmentParams, Observation, RewardUtils
from f1tenth_gym.envs.reward import Reward


class CenterReward(Reward):
    def __init__(self, params: dict):
        self.params = EnvironmentParams(params)
        self.center_gain = 0.5
        self.steering_gain = 1.0
        self.center_threshold = 0.25

    def reset(self, params: dict):
        # reset values on crash/lap finish
        self.params = EnvironmentParams(params)

    def reward(self, obs, action):
        obs = Observation(obs)
        # get action for agent 0
        steer, _ = (action[0][0], action[0][1])
        base_reward = 1.0

        center_dist = RewardUtils.center_distance(obs, self.params)
        center_dist = 0 if center_dist < self.center_threshold else center_dist

        reward = 0.01 * (
            base_reward
            - self.center_gain * abs(center_dist)
            - self.steering_gain * abs(steer)
        )

        # clamp reward [0, inf)
        reward = max(reward, 0.0)

        return reward, False
