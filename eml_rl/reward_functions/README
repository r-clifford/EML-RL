# Reward Template User Guide

## Motivation
The purpose of a reward function is to incentivize the model to perform actions that we want. It does this by giving the model a reward based on an observation that the model makes in the environment. The model then attempts to maximize its reward by changing its behavior. By maximizing its reward, we _reinforce_ behaviors that we want the model to do. This is called **_reinforcement learning_**.

## How it Works
 For example, say we want to incentivize the model to complete laps around the track. An extremely simple way of doing this (quite inefficiently) is to simply reward the model based on its progress around the track. This method takes a long time to train, and is not very good compared to other methods. We could expand this principle by scaling the reward based on the car's speed. This would provide a larger reward for going through the track faster. 

 ## Making Your Own 

The reward function's structure is quite simple. An Observation is input to the function, accessible through the variable ```obs```. Some code is written to compute the reward, and the reward is returned, along with a boolean to indicate a reset to the run. For example, if we wanted to return a constant reward of ```1```, we would write the following reward function:
```python
def reward(self, obs, action):
        # use observation variable (obs) to get the information
        # see Observation class in reward.py
        obs = Observation(obs)
        #return reward,reset
        return 1, False
```
This function will return a constant reward of ```1``` and will not reset the car to its initial state upon returning from the function, which is what we want.

To elaborate on this, a simple reward function to emphasize lap progress can be made by simply returning ```progress```. The field ```progress``` is contained in the function's input, so accessing it is simple. An example reward function is detailed below:
```python
def reward(self, obs, action):
        # calculate reward based on progress through the track
        # use observation variable (observation) to get the information
        # see Observation class in reward.py for more detailed information
        obs = Observation(obs)
        progress = obs.lap_progress
        return progress, False
```
The main difference between this reward function and the previous reward function is the introduction of a variable return value. A model has been trained on this reward function and was able to go around the track at slow speeds after 30 minutes or so.

There are many different accessible parameters in the function's input, ```obs```. This variable is an instance of the ```Observation``` class, which is shown below with some useful variables commented:
```python
class Observation:
    # class for observation information
    ang_vels_z: float #angular velocity in the z direction
    collisions: float #number of collisions
    ego_idx: float
    lap_counts: float #number of laps completed
    lap_progress: float #progress through the track
    lap_times: float #times for completed laps
    linear_vels_x: float #linear x velocity
    linear_vels_y: float #linear y velocity
    poses_theta: float
    poses_x: float
    poses_y: float
    scans: np.ndarray[float] #lidar scans

    #initialize an instance of Observation
    def __init__(self,input_obs: dict):
        self.ang_vels_z = input_obs['ang_vels_z'][0]
        self.collisions = input_obs['collisions'][0]
        self.ego_idx = input_obs['ego_idx']
        self.lap_counts = input_obs['lap_counts'][0]
        self.lap_progress = input_obs['lap_progress'][0]
        self.linear_vels_x = input_obs['linear_vels_x'][0]
        self.linear_vels_y = input_obs['linear_vels_y'][0]
        self.poses_theta = input_obs['poses_theta'][0]
        self.poses_x = input_obs['poses_x'][0]
        self.poses_y = input_obs['poses_y'][0]
        self.scans = input_obs['scans'][0]
```

## Try it Yourself!
Try creating a new reward function to make the car go around the track. Use a combination of variables in the ```Observation``` class to give a value for the reward based on how you would like the model to act. Use the following commands to make a copy of ```TemplateReward.py``` for you to use:
```shell
# Commands run from directory EML_RL
cp ./eml_rl/reward_functions/TemplateReward.py ./eml_rl/reward_functions/MyReward.py
```
Change the configuration file to use your reward function. Edit ```configuration.py``` by navigating to ```eml_rl/```. In ```configuration.py```, add the following line under the other imports:
```python
from eml_rl.reward_functions.MyReward import MyReward
```
Then, change the ```reward_function``` to be the name of your reward function class (MyReward) to look like the following line:
```python
reward_function = MyReward
```
Open your text editor of choice and edit the reward class in ```MyReward.py```. Change the name of the class from ```TemplateReward``` to ```MyReward```. The declaration at the top of the class on line ```5``` should now look like
```python
class MyReward(Reward):
```
Now, you can make your reward function. A trivial function rewarding solely progress is there as an example. Feel free to elaborate on this function, or come up with a better method!

## Train Your Model Using Your Reward Function
Once you have edited the reward function to your liking, it is time to train! Since you will already have edited ```configuration.py``` to use your specific reward function, training using the function is simple. Follow the commands below after following the [initial installation instructions](https://github.com/r-clifford/EML-RL/tree/reward-template) explained in the README to begin training your model:
```shell
#run these commands from the directory EML_RL
source rl_venv/bin/activate
# make a logs directory if it has not been already created
mkdir logs
# use logs as the <log_dir>
# this command is run from the base directory of the repository
# example format: ./eml_rl/train.sh <log_dir> <algorithm> <config_file>
# below is an example command to run the td3 algorithm using your reward function
./eml_rl/train.sh logs/ td3 ./eml_rl/config/hyperparams/td3_f1tenth.py
```

## Evaluate Your Trained Model
After training for awhile (times may vary based on the hardware you are running on), you will want to see how your model is doing. Kill the training session with Ctrl-c. You should get a message like the one below. Copy this message (we will use it in a second)
```
Saving to logs/td3-1738348143/td3/f1tenth-v0_1_eb2c0f96-bbf0-4d46-a0e4-03eb760dc3fd
```
It is useful (and fun) to watch your model drive. Some commands are listed below to begin an evaluation session on the model you just trained:
```shell
# Assuming we trained using td3, change this if you used something different
# python3 eml_rl/eval.py <algorithm> <path to model zip>
#example command shown below, find your path to a file best_model.zip under your training session in logs
#example command, do not attempt to run
#find your own best_model.zip in your training logs
#python3 eml_rl/eval.py td3 ./logs/td3-1738348143/td3/f1tenth-v0_1_eb2c0f96-bbf0-4d46-a0e4-03eb760dc3fd/best_model.zip
```




