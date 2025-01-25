from eml_rl.reward import EnvironmentParams, Observation
from f1tenth_gym.envs.reward import Reward


class TemplateReward(Reward):
    max_progress: float
    def __init__(self,params:dict):
        self.params = EnvironmentParams(params)
        self.best_progress = 0

    def reset(self):
        #reset values on crash/lap finish
        pass

    def reward(self, obs, action):
        #calculate reward based on progress through the track
        #use observation variable (observation) to get the information
        #see Observation class in reward.py for more detailed information
        obs = Observation(obs)
        progress = obs.lap_progress
        return progress, False