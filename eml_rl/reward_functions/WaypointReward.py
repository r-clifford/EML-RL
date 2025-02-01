from eml_rl.reward import EnvironmentParams, Observation, RewardUtils
from f1tenth_gym.envs.reward import Reward
import math


class WaypointReward(Reward):
    def __init__(self, params: dict):
        self.params = EnvironmentParams(params)
        self.lookahead = 1.0
        self.cd_max = 1.0

    def reset(self, params: dict):
        # reset values on crash/lap finish
        self.params = EnvironmentParams(params)

    def reward(self, obs, action):
        observation = Observation(obs)
        # get action for agent 0
        action = action[0]
        steer, speed = (action[0], action[1])

        # get nearest waypoint
        waypoint = RewardUtils.nearest_waypoint(
            observation, self.params, "race")
        _, _, x, y, yaw, kappa, v, _ = waypoint

        # Reduce target speed
        v *= 0.5

        dist_from_waypoint = math.sqrt(
            (x - observation.poses_x) ** 2 + (y - observation.poses_y) ** 2
        )
        # deviation from waypoints curvature
        curv_deviation = abs(observation.poses_theta - yaw)

        # steering deviation (unused)
        st_deviation = abs(steer - kappa)  # noqa: F841

        # velocity deviation
        v_deviation = abs(
            speed - max(min(v, self.params.v_max), self.params.v_min))

        reward = (
            1
            - 0.25 * dist_from_waypoint
            - curv_deviation / self.cd_max
            - v_deviation / (self.params.v_max - self.params.v_min)
        )
        reward *= 0.01

        return reward, False
