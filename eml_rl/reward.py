import numpy as np
from f1tenth_planning.control.pure_pursuit.pure_pursuit import PurePursuitPlanner
from dataclasses import dataclass

from f1tenth_gym.envs.track import Track
import f1tenth_gym.envs.track.utils as track_utils
import math


@dataclass
class Observation:
    # class for observation information
    ang_vels_z: float
    collisions: float
    ego_idx: float
    lap_counts: float
    lap_progress: float
    lap_times: float
    linear_vels_x: float
    linear_vels_y: float
    poses_theta: float
    poses_x: float
    poses_y: float
    scans: np.ndarray[float]

    def __init__(self, input_obs):
        self.ang_vels_z = input_obs["ang_vels_z"][0]
        self.collisions = input_obs["collisions"][0]
        self.ego_idx = input_obs["ego_idx"]
        self.lap_counts = input_obs["lap_counts"][0]
        self.lap_progress = input_obs["lap_progress"][0]
        self.linear_vels_x = input_obs["linear_vels_x"][0]
        self.linear_vels_y = input_obs["linear_vels_y"][0]
        self.poses_theta = input_obs["poses_theta"][0]
        self.poses_x = input_obs["poses_x"][0]
        self.poses_y = input_obs["poses_y"][0]
        self.scans = input_obs["scans"][0]


class EnvironmentParams:
    v_max: float
    v_min: float
    map: str
    num_agents: float
    model: str
    max_laps: float
    track: Track

    def __init__(self, params: dict):
        self.v_max = params["params"]["v_max"]
        self.v_min = params["params"]["v_min"]
        self.map = params["map"]
        self.num_agents = params["num_agents"]
        self.model = params["model"]
        self.max_laps = params["max_laps"]
        self.track = params["track"]


class RewardUtils:
    @staticmethod
    def center_distance(obs: Observation, params: EnvironmentParams) -> float:
        ego_x, ego_y = obs.poses_x, obs.poses_y
        centerline = params.track.centerline
        pairs = np.dstack((centerline.xs, centerline.ys))[0]
        _, _, _, index = track_utils.nearest_point_on_trajectory(
            np.array([ego_x, ego_y]), pairs
        )
        center_x, center_y = (centerline.xs[index], centerline.ys[index])

        dist = math.sqrt((ego_x - center_x) ** 2 + (ego_y - center_y) ** 2)
        return dist
