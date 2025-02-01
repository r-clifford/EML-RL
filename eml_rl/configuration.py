from eml_rl.reward_functions.ProgressReward import ProgressReward
from eml_rl.reward_functions.ScaledReward import ScaledReward
from eml_rl.reward_functions.CenterReward import CenterReward
from eml_rl.reward_functions.WaypointReward import WaypointReward
from eml_rl.reward_functions.PurePursuitReward import PurePursuitReward


# Max and min velocity in m/s
# Ensure vmin > 0.0 to avoid stopping/reversing
# Higher vmax may allow better laptimes, though training is often
# significantly slower as model crashes a lot more
vmax = 4.0
vmin = 1.0

# Number of lidar observations to stack
# May allow for model to learn temporal understanding
# as the lidar scans a t = 0 -> t = `frame_stack` are concatenated
# Keep in mind that a scan is retrieved every ~30 ms
# Recommended 5-40
frame_stack = 40

# Number of lidar beams to sample
# Unmodified scan is 1080 beams
# We linearly sample `lidar_beams` from the full scan
# Recommended on of [20, 40, 80]
lidar_beams = 80

# Reward function to use
# Available reward functions in `git_root`/eml_rl/reward_functions
# Make sure to import requested function
# Example:
# from eml_rl.reward_functions.ProgressReward import ProgressReward
reward_function = CenterReward

# cl_grid_static: centerline, fixed start point
# cl_random_static: centerline, random
# shuf_random_static: raceline
reset_config = "cl_random_static"
