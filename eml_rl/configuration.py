from eml_rl.reward_functions.ProgressReward import ProgressReward
from eml_rl.reward_functions.ScaledReward import ScaledReward

vmax = 4.0
vmin = 1.0
frame_stack = 5
lidar_beams = 80

reward_function = ProgressReward