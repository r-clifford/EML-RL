from copy import deepcopy
import numpy as np
from f1tenth_planning.control.pure_pursuit.pure_pursuit import PurePursuitPlanner
import math
import time

from f1tenth_gym.envs.reward import Reward


class ProgressReward(Reward):
    def __init__(self):
        self.buffer_size = 30
        self.prog_buff = np.zeros(self.buffer_size)
        self.window_size_rev = 3

    def reset(self):
        self.prog_buff = np.zeros(self.buffer_size)

    def reward(self, obs, action):
        progress = obs["lap_progress"][0]

        if obs["collisions"][0] > 0.1:
            return -1, True
        if progress - self.prog_buff[0] > 0.5:
            return -2, True
        if progress < self.prog_buff[1 : self.window_size_rev].mean():
            if self.prog_buff.mean() < 0.97:
                return -2, True
            else:
                return 2, True
        agent_speed = action[0][1]
        agent_steer = action[0][0]

        self.prog_buff = np.roll(self.prog_buff, 1)
        self.prog_buff[0] = progress
        # prog_diff = self.prog_buff[0] - self.prog_buff[-1]
        short_term = self.prog_buff[0] - self.prog_buff[-20]
        med_term = self.prog_buff[0] - self.prog_buff[-10]
        long_term = self.prog_buff[0] - self.prog_buff[-1]
        if progress < 0.1 and abs(agent_steer) > 0.2:
            return -0.001, False

        prog_reward = 2.5 * short_term + 7.5 * med_term + 10.0 * long_term
        # bonus = (progress < 0.1) * 5
        bonus = 1.0
        speed_reward = 0.01 * agent_speed

        reward = prog_reward * bonus + speed_reward

        reward = max(reward, 0.0)
        return reward, False
