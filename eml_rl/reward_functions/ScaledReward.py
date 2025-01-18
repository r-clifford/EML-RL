from eml_rl.reward import EnvironmentParams, Observation
from f1tenth_gym.envs.reward import Reward



class ScaledReward(Reward):
    def __init__(self,params:dict):
        self.params = EnvironmentParams(params)

    def reset(self):
        #reset values on crash/lap finish
        pass

    def reward(self, obs, action):
        observation = Observation(obs)
        #get action for agent 0
        action = action[0]
        
        return 0,False