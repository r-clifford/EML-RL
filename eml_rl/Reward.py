from copy import deepcopy
import numpy as np
from f1tenth_planning.control.pure_pursuit.pure_pursuit import PurePursuitPlanner
import math
import time

from f1tenth_gym.envs.reward import Reward


class ProgressReward(Reward):
    def __init__(self):
        self.last_progress = 0
        self.buffer_size = 30
        self.prog_buff = np.zeros(self.buffer_size)

    def reward(self, obs, action):
        progress = obs["lap_progress"][0]
        if obs["collisions"][0] > 0.1:
            self.last_progress = 0
            self.prog_buff = np.zeros(self.buffer_size)
            return -1
        if progress - self.last_progress > 0.5:
            return -1
        if self.last_progress > progress:
            print(self.last_progress, progress)
            return -0.1
        self.last_progress = progress
        agent_speed = action[0][1]
        agent_steer = action[0][0]

        self.prog_buff = np.roll(self.prog_buff, 1)
        self.prog_buff[0] = progress
        # prog_diff = self.prog_buff[0] - self.prog_buff[-1]
        short_term = self.prog_buff[0] - self.prog_buff[-20]
        med_term = self.prog_buff[0] - self.prog_buff[-10]
        long_term = self.prog_buff[0] - self.prog_buff[-1]
        if progress < 0.1 and abs(agent_steer) > 0.2:
            return -0.001

        prog_reward = 2.5 * short_term + 7.5 * med_term + 10.0 * long_term
        # bonus = (progress < 0.1) * 5
        bonus = 1.0
        speed_reward = 0.01 * agent_speed

        reward = prog_reward * bonus + speed_reward


        reward = max(reward, 0.0)
        return reward
        # progress = obs["lap_progress"][0]
        # if obs["collisions"][0] > 0.1:
        #     self.last_progress = 0
        #     self.prog_buff = np.zeros(10)
        #     return -1
        # if progress - self.last_progress > 0.5:
        #     return -1
        # if self.last_progress > progress:
        #     print(self.last_progress, progress)
        # self.last_progress = progress
        # agent_speed = action[0][1]
        # agent_steer = action[0][0]
        #
        # self.prog_buff = np.roll(self.prog_buff, 1)
        # self.prog_buff[0] = progress
        # prog_diff = self.prog_buff[0] - self.prog_buff[-1]
        # if (progress < 0.1 and abs(agent_steer) > 0.2):
        #     return -0.001
        #
        # reward = (
        #     20.0 * prog_diff + 0.03 * agent_speed
        # )
        # reward = max(reward, 0.0)
        # return reward
        #
        #


# def RewardFunction(obs, action, target_steer, target_speed, avg_steer):
#     # Separate into imitation rewards and extrinsic rewards
#     target_speed = 0.7 * np.clip(target_speed, 0.0, 8.0)  # *0.7
#     target_steer = np.clip(target_steer, -0.4189, 0.4189)
#     beta_c = 0.4  # 0.84#0.4
#     # beta_steer_weight = 0.4
#     beta_steer_weight = 0.39
#     beta_velocity_weight = 0.4
#
#     # max_steer_diff = 0.8
#     max_steer_diff = 1.5
#     max_velocity_diff = 1.5
#     # if prev_obs is None: return 0
#     observation = deepcopy(obs)["agent_0"]
#     # return observation["linear_vel_x"] *0.1
#     # if self.lap_count < observation["lap_count"]:
#     #     self.lap_count = observation["lap_count"]
#     #     return 1 + 3.0 * math.exp(9-observation["lap_time"]/7)  # complete
#     if observation["collision"] > 0.1:
#         return -1  # + observation["progress"] # temp -10 from -1
#     # action = self.agent.plan(observation)
#     agent_steer = action[0][0]
#     agent_speed = action[0][1]
#     # print(f"{target_steer} {target_speed} {agent_steer} {agent_speed}")
#     steer_reward = (
#         abs(target_steer - agent_steer) / max_steer_diff
#     ) * beta_steer_weight
#     throttle_reward = (
#         abs(target_speed - agent_speed) / max_velocity_diff
#     ) * beta_velocity_weight
#     # if (target_speed - 5.6 <= 0.001): # allow potentially faster speeds
#     #     throttle_reward = 0 if agent_speed > 5.6 else throttle_reward
#
#     reward = beta_c - steer_reward - throttle_reward
#
#     reward *= 0.5
#     # reward = (reward if agent_speed > 0.1 else 0)
#
#     # avg_steer = np.array(list(self.steer_history.queue)).mean()
#     steer_diff = abs(agent_steer - avg_steer)
#     steer_penalty = 0.00002 * abs(agent_steer)
#     # self.steer_history.get()
#     # self.steer_history.put(agent_steer)
#     # print(reward)
#     # steer_reward = 0.1 if abs(agent_steer) < 0.1 else 0
#     # reward = (0.1*observation["lap_time"] + steer_reward if (observation["linear_vel_x"] > 0.1) else 0)#- 0.25
#     # reward = reward if observation["linear_vel_x"] > 0.1 else -0.001
#     # print(f"{obs['agent_0']['lap_time']:4f} : st: {target_steer:3f}  | {agent_steer:3f} sp: {target_speed:3f} | {agent_speed:3f} r: {reward:3f}")
#     reward = reward - steer_diff * 0.01 - steer_penalty
#     reward = 0.75 * max(reward, 0.00)  # limit at 0
#     return reward


def RewardFunction(obs, action, target_steer, target_speed, avg_steer):
    target_speed = 0.9 * np.clip(target_speed, 0.0, 8.0)  # *0.7
    max_velocity_diff = 1.5
    beta_velocity_weight = 0.4
    # observation = deepcopy(obs)["agent_0"]
    if obs["agent_0"]["collision"] > 0.1:
        return -1  # + observation["progress"] # temp -10 from -1
    agent_steer = action[0][0]
    agent_speed = action[0][1]
    steer_diff = abs(agent_steer - avg_steer)
    steer_penalty = 0.0001 * math.pow(abs(agent_steer), 4)
    throttle_reward = (
        abs(target_speed - agent_speed) / max_velocity_diff
    ) * beta_velocity_weight

    reward = 0.4 - throttle_reward
    # reward = reward - steer_diff * 0.03 - steer_penalty
    reward = reward - steer_diff * 0.05 - steer_penalty
    reward = 0.75 * max(reward, 0.00)  # limit at 0
    return reward


# def RewardFunction(obs, action, target_steer, target_speed, avg_steer):
#     observation = deepcopy(obs)["agent_0"]
#     scan = observation["scan"]
#
#     max_idx = np.argmax(scan)
#     print(max_idx)
#     max_val = max_idx / len(scan)
#     # print(scan)
#     diff = abs(max_val - 0.5)
#     reward = 0.01 - 0.02 * diff
#     reward = max(reward, 0.00)  # limit at 0
#     agent_speed = action[0][1]
#     # reward += 0.001 * agent_speed
#     reward += 0.0001 * observation["linear_vel_x"]
#     return reward
