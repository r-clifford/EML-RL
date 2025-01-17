from copy import deepcopy
import numpy as np
from f1tenth_planning.control.pure_pursuit.pure_pursuit import PurePursuitPlanner
import math
import time
from dataclasses import dataclass
import pprint

from f1tenth_gym.envs.reward import Reward

#data format for observation dict
# {'ang_vels_z': array([0.], dtype=float32),
#  'collisions': array([0.], dtype=float32),
#  'ego_idx': 0,
#  'lap_counts': array([0.], dtype=float32),
#  'lap_progress': array([0.], dtype=float32),
#  'lap_times': array([0.], dtype=float32),
#  'linear_vels_x': array([0.], dtype=float32),
#  'linear_vels_y': array([0.], dtype=float32),
#  'poses_theta': array([1.0791948], dtype=float32),
#  'poses_x': array([-8.878114], dtype=float32),
#  'poses_y': array([11.707165], dtype=float32),
#  'scans': array([[1.2277151, 1.2545907, 1.2332468, ..., 2.2302256, 7.6173387,
#         7.631844 ]], dtype=float32)}
@dataclass
class obs:
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
    def __init__(self,input_obs):
        self.ang_vels_z = input_obs['ang_vels_z'][0]
        self.collisions = input_obs['collisions'][0]
        self.ego_idx = input_obs['ego_idx'][0]
        self.lap_counts = input_obs['lap_counts'][0]
        self.lap_progress = input_obs['lap_progress'][0]
        self.linear_vels_x = input_obs['linear_vels_x'][0]
        self.linear_vels_y = input_obs['linear_vels_y'][0]
        self.poses_theta = input_obs['poses_theta'][0]
        self.poses_x = input_obs['poses_x'][0]
        self.poses_y = input_obs['poses_y'][0]
        self.scans = input_obs['scans'][0]

#data format for params dict
# ({'config': {'params_randomizer': <function randomize_sim_params.<locals>.f at 0x7f231d275240>,
#  'params': {'mu': 0.3, 'v_max': 4.0, 'v_min': 1.0},
#  'reset_config': {'type': 'shuf_random_static'},
#  'reward_class': <class 'eml_rl.reward.ScaledReward'>,
#  'map': 'Oschersleben', 
#  'num_agents': 1,
#  'timestep': 0.01,
#  'model': 'st',
#  'control_input': ['speed', 'steering_angle'],
#  'observation_config': {'type': 'features','features': ['pose_x', 'pose_y', 'scan','pose_theta', 'linear_vel_x', 'ang_vel_z', 'collision', 'lap_time', 'lap_count']},'max_laps': 3}, 'render_mode': 'rgb_array'})
class ScaledReward(Reward):
    config: function
    mu: float
    v_max: float
    v_min: float
    reset_config: type
    reward_class: dataclass
    map: str
    num_agents: float
    timestep: float
    model: str
    speed: float
    steering_angle: float
    max_laps: float
    render_mode: str
    def __init__(self, params: dict):
        self.config = params['config']['params_randomizer']
        self.mu = params['params']['mu']
        self.v_max = params['params']['v_max']
        self.v_min = params['params']['v_min']
        self.reset_config = params['reset_config']['type']
        self.reward_class = params['reward_class']
        self.map = params['map']
        self.num_agents = params['num_agents']
        self.timestep = params['timestep']
        self.model = params['model']
        self.speed = params['control_input']['speed']
        self.steering_angle = params['control_input']['steering_angle']
        self.max_laps = params['max_laps']
        self.render_mode = params['render_mode']

    def reset(self):
        #reset values on crash/lap finish
        pass

    def reward(self, obs, action):
        
        return 0,False
