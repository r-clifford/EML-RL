from eml_rl.reward import EnvironmentParams, Observation, RewardUtils, LineType
from f1tenth_gym.envs.reward import Reward
import math


class WaypointReward(Reward):
    def __init__(self, params: dict):
        self.params = EnvironmentParams(params)
        self.lookahead = 1.0
        self.cd_max = 0.3
        self.count = -1

    def reset(self, params: dict):
        # reset values on crash/lap finish
        self.params = EnvironmentParams(params)
        self.count = -1

    def reward(self, obs, action):
        observation = Observation(obs)
        # get action for agent 0
        action = action[0]
        steer, speed = (action[0], action[1])

        # get nearest waypoint
        waypoint = RewardUtils.nearest_waypoint(
            observation, self.params, LineType.Raceline
        )
        _, _, x, y, yaw, kappa, v, _ = waypoint

        dist_from_waypoint = math.sqrt(
            (x - observation.poses_x) ** 2 + (y - observation.poses_y) ** 2
        )
        # deviation from waypoints curvature
        theta = observation.poses_theta - math.pi / 2
        curv_deviation = abs(theta - yaw)

        # steering deviation (unused)
        st_deviation = abs(steer - kappa)  # noqa: F841

        # velocity deviation
        v_deviation = abs(speed - max(min(v, self.params.v_max), self.params.v_min))

        # reward = (
        #     1
        #     - 0.25 * dist_from_waypoint
        #     - v_deviation / (self.params.v_max - self.params.v_min)
        # )
        reward = (
            1
            * (min(1.0, 0.1 / (dist_from_waypoint + 1e-12)))
            * (
                min(
                    1.0, (self.params.v_max - self.params.v_min) / (v_deviation + 1e-12)
                )
            )
        )
        reward *= 0.01

        if observation.collisions > 0:
            reward = -0.001

        return reward, False
