from copy import deepcopy
import numpy as np
from f1tenth_planning.control.pure_pursuit.pure_pursuit import PurePursuitPlanner
def RewardFunction(obs, action, target_steer, target_speed, avg_steer):
    target_speed = 0.9*np.clip(target_speed, 0.0, 8.0) # *0.7
    target_steer = np.clip(target_steer, -0.4189,0.4189)
    beta_c = 0.4#0.84#0.4
    beta_steer_weight = 0.4
    beta_velocity_weight = 0.4

    max_steer_diff = 0.8
    max_velocity_diff = 1.5
    # if prev_obs is None: return 0
    observation = deepcopy(obs)["agent_0"]
    # return observation["linear_vel_x"] *0.1
    # if self.lap_count < observation["lap_count"]:
    #     self.lap_count = observation["lap_count"]
    #     return 1 + 3.0 * math.exp(9-observation["lap_time"]/7)  # complete
    if observation['collision']>0.1:
        return -1# + observation["progress"] # temp -10 from -1
    # action = self.agent.plan(observation)
    agent_steer = action[0][0]
    agent_speed = action[0][1] 
    # print(f"{target_steer} {target_speed} {agent_steer} {agent_speed}")
    steer_reward =  (abs(target_steer - agent_steer) / max_steer_diff)  * beta_steer_weight
    throttle_reward =   (abs(target_speed - agent_speed) / max_velocity_diff) * beta_velocity_weight
    # if (target_speed - 5.6 <= 0.001): # allow potentially faster speeds
    #     throttle_reward = 0 if agent_speed > 5.6 else throttle_reward

    reward = beta_c - steer_reward - throttle_reward
    reward = max(reward, 0.00) # limit at 0

    reward *= 0.5
    # reward = (reward if agent_speed > 0.1 else 0)

    # avg_steer = np.array(list(self.steer_history.queue)).mean()
    steer_diff = abs(agent_steer - avg_steer)
    # self.steer_history.get()
    # self.steer_history.put(agent_steer)
    # print(reward)
    # steer_reward = 0.1 if abs(agent_steer) < 0.1 else 0
    # reward = (0.1*observation["lap_time"] + steer_reward if (observation["linear_vel_x"] > 0.1) else 0)#- 0.25
    # reward = reward if observation["linear_vel_x"] > 0.1 else -0.001
    # print(f"{obs['agent_0']['lap_time']:4f} : st: {target_steer:3f}  | {agent_steer:3f} sp: {target_speed:3f} | {agent_speed:3f} r: {reward:3f}")
    return reward