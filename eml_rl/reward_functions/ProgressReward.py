from copy import deepcopy
import numpy as np
from f1tenth_planning.control.pure_pursuit.pure_pursuit import PurePursuitPlanner
import math
import time

from f1tenth_gym.envs.reward import Reward


class ProgressReward(Reward):
    def __init__(self, params: dict):
        self.params = params
        self.buffer_size = 30
        self.prog_buff = np.zeros(self.buffer_size)
        self.window_size_rev = 3
        self.steer_buff = np.zeros(2)
        self.sc = 2.5
        self.mc = 7.5
        self.lc = 10.0
        self.sr = 0.01
        self.sp = 0.05
        self.steer_weight = 0.00
        self.speed_weight = 0.00

        self.s_max = params["params"]["s_max"]
        self.s_min = params["params"]["s_min"]
        self.best_prog = 0.0
        # See __speed
        self.laps = 0
        self.laps_to_max_speed = 100
        self.init_speed_target = 3.0
        self.max_speed = params["params"]["v_max"]
        self.min_speed = params["params"]["v_min"]

    def reset(self):
        self.prog_buff = np.zeros(self.buffer_size)
        self.steer_buff = np.zeros(2)

    def __progress(self, action: tuple[float, float], obs: dict) -> tuple[float, bool]:
        """Calculate base reward from progress

        Args:
            action (steer, speed)
            obs (observation)

        Returns:
            (reward, done)
        """
        agent_steer, _ = action
        progress = obs["lap_progress"][0]
        min_reward = -1.0
        if obs["collisions"][0] > 0.1:
            return min_reward, True
        if progress - self.prog_buff[0] > 0.5:
            return 2 * min_reward, True
        if progress < self.prog_buff[1 : self.window_size_rev].mean():
            if self.prog_buff.mean() < 0.97:
                return 2 * min_reward, True
            else:
                self.laps += 1
                self.best_prog = 1.0
                return 2, True
        # if self.prog_buff[0] - progress > 0.5:
        #     return 2, True
        self.prog_buff = np.roll(self.prog_buff, 1)
        self.prog_buff[0] = progress
        short_term = self.prog_buff[0] - self.prog_buff[-20]
        med_term = self.prog_buff[0] - self.prog_buff[-10]
        long_term = self.prog_buff[0] - self.prog_buff[-1]
        prog_reward = self.sc * short_term + self.mc * med_term + self.lc * long_term
        self.best_prog = max(self.best_prog, progress)

        if progress < 0.1:
            if abs(agent_steer) > self.s_max / 2:
                return -0.01, False
        # else:
        #     return max(prog_reward, 0.001), False

        return prog_reward, False  # + min(progress / 10, 0.02), False

    def reward(self, obs, action):
        agent_speed = action[0][1]
        agent_steer = action[0][0]

        # Calculate base reward from lap progress
        prog_reward, done = self.__progress(action[0], obs)

        # Scale reward based on speed
        speed_mod = 1 - self.__speed(agent_speed)
        # Scale reward based on steering input
        steer_mod = max(1 - self.__steer(agent_steer), 0.0)
        reward = prog_reward * steer_mod * speed_mod
        # print(reward)
        # if steer_mod < 0.5:
        #     print(action)
        #     print(self.steer_buff)
        #     print(steer_mod)
        #     print(reward)
        #     print("")
        #     breakpoint()
        # reward += agent_speed * 0.01
        reward = max(reward, 0.0)
        return reward, done

    def __steer(self, agent_steer: float) -> float:
        """Compute reward modifier related to steering input

        Args:
            agent_steer: float

        Returns:
            Computed reward modifier
        """
        power = 2.0
        if len(self.steer_buff) >= 2:
            self.steer_buff = np.roll(self.steer_buff, 1)
            self.steer_buff[0] = agent_steer
        else:
            self.steer_buff.fill(agent_steer)

        max_diff = self.s_max - self.s_min
        signs = np.abs(np.sign(self.steer_buff).sum())
        steer_diff = np.abs(self.steer_buff[0] - self.steer_buff[1])
        # self.steer_weight *
        return self.steer_weight * (
            (steer_diff * 2) ** power / max_diff**power + signs / 4.0
        )

    def __speed(self, speed: float) -> float:
        """Compute reward modifier related to speed input
        Discourage exceeding a specific speed until substantial progress is made
        Increase the max speed as laps are completed
        Args:
            speed: float

        Returns:
            Computed reward modifier
        """
        target_max_speed = self.laps / self.laps_to_max_speed * self.max_speed
        target_max_speed = max(target_max_speed, self.min_speed)
        if speed > target_max_speed:
            return max(
                0, (speed - target_max_speed) / (self.max_speed - target_max_speed)
            )
        else:
            return 0
        # return self.speed_weight * (1 - speed / self.params["params"]["v_max"])