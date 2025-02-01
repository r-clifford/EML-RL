from f1tenth_gym.envs.track.raceline import Raceline
import numpy as np
from dataclasses import dataclass

from f1tenth_gym.envs.track import Track
import f1tenth_gym.envs.track.utils as track_utils
import math
from enum import Enum, auto


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
        self.s_min = params["params"]["s_min"]
        self.s_max = params["params"]["s_max"]
        self.map = params["map"]
        self.num_agents = params["num_agents"]
        self.model = params["model"]
        self.max_laps = params["max_laps"]
        self.track = params["track"]


class LineType(Enum):
    Raceline = auto()
    Centerline = auto()


class RewardUtils:
    @staticmethod
    def nearest_waypoint_index(obs: Observation, line: Raceline) -> int:
        ego_x, ego_y = obs.poses_x, obs.poses_y
        pairs = np.dstack((line.xs, line.ys))[0]
        _, _, _, index = track_utils.nearest_point_on_trajectory(
            np.array([ego_x, ego_y]), pairs
        )
        return index

    @staticmethod
    def _line_distance(obs: Observation, line: Raceline) -> float:
        ego_x, ego_y = obs.poses_x, obs.poses_y
        index = RewardUtils.nearest_waypoint_index(obs, line)
        center_x, center_y = (line.xs[index], line.ys[index])
        dist = math.sqrt((ego_x - center_x) ** 2 + (ego_y - center_y) ** 2)
        return dist

    @staticmethod
    def center_distance(obs: Observation, params: EnvironmentParams) -> float:
        return RewardUtils._line_distance(obs, params.track.centerline)

    @staticmethod
    def raceline_distance(obs: Observation, params: EnvironmentParams) -> float:
        return RewardUtils._line_distance(obs, params.track.raceline)

    @staticmethod
    def nearest_waypoint(
        obs: Observation,
        params: EnvironmentParams,
        line_type: LineType,
    ) -> tuple[int, float, float, float, float, float, float, float]:
        """Get waypoint closest to current position
        Note: centerline will return s, yaw, k, ax = 0

        Args:
            obs:
            params:
            line_type: LineType.{Centerline, Raceline}

        Returns:
            (index, s, x, y, yaw, kappa, v, a)
        """
        line = (
            params.track.raceline
            if line_type == LineType.Raceline
            else params.track.centerline
        )
        index = RewardUtils.nearest_waypoint_index(obs, line)
        s = 0
        yaw = 0
        k = 0
        ax = 0
        if line_type == LineType.Raceline:
            s = line.ss[index]
            yaw = line.yaws[index]
            k = line.ks[index]
            ax = line.axs[index]
        x = line.xs[index]
        y = line.ys[index]
        vx = line.vxs[index]
        return (index, s, x, y, yaw, k, vx, ax)
