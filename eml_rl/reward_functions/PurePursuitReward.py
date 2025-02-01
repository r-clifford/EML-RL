from eml_rl.reward import EnvironmentParams, Observation
from f1tenth_gym.envs.reward import Reward
from f1tenth_planning.control.pure_pursuit import pure_pursuit
import numpy as np


class PurePursuitReward(Reward):
    def __init__(self, params: dict):
        self.params = EnvironmentParams(params)
        self.lookahead = 1.0
        self.cd_max = 1.0
        raceline = self.params.track.raceline
        waypoints = np.stack(
            [
                raceline.xs,
                raceline.ys,
                raceline.vxs,
                raceline.yaws,
            ],
            axis=1,
        )

        self.controller = pure_pursuit.PurePursuitPlanner(
            params["params"]["length"], waypoints=waypoints
        )

    def reset(self, params: dict):
        # reset values on crash/lap finish
        self.params = EnvironmentParams(params)

        raceline = self.params.track.raceline
        waypoints = np.stack(
            [
                raceline.xs,
                raceline.ys,
                raceline.vxs,
                raceline.yaws,
            ],
            axis=1,
        )
        self.controller = pure_pursuit.PurePursuitPlanner(
            params["params"]["length"], waypoints=waypoints
        )

    def reward(self, obs, action):
        observation = Observation(obs)
        # get action for agent 0
        action = action[0]
        steer, speed = (action[0], action[1])

        target_steer, target_speed = self.controller.plan(
            observation.poses_x,
            observation.poses_y,
            observation.poses_theta,
            self.lookahead,
        )

        target_speed *= 0.5

        delta_steer = abs(target_steer - steer)
        delta_speed = abs(min(self.params.v_max, target_speed) - speed)

        reward = 1.0 - (
            delta_speed / (self.params.v_max) + delta_steer / self.params.s_max
        )
        reward *= 0.01

        return reward, False
