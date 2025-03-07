import importlib
# from eml_rl.reward_functions.ProgressReward import ProgressReward
# from eml_rl.reward_functions.ScaledReward import ScaledReward
# from eml_rl.reward_functions.CenterReward import CenterReward
# from eml_rl.reward_functions.WaypointReward import WaypointReward
# from eml_rl.reward_functions.PurePursuitReward import PurePursuitReward
import json

# Read the configuration.json file
with open("eml_rl/configuration.json", "r") as file:
    config_file = json.load(file)

# Max and min velocity in m/s
# Ensure vmin > 0.0 to avoid stopping/reversing
# Higher vmax may allow better laptimes, though training is often
# significantly slower as model crashes a lot more
vmax = config_file["vmax"]
vmin = config_file["vmin"]

# Number of lidar observations to stack
# May allow for model to learn temporal understanding
# as the lidar scans a t = 0 -> t = `frame_stack` are concatenated
# Keep in mind that a scan is retrieved every ~30 ms
# Recommended 5-40
frame_stack = config_file["frame_stack"]
frame_skip = config_file["frame_skip"]

# Number of lidar beams to sample
# Unmodified scan is 1080 beams
# We linearly sample `lidar_beams` from the full scan
# Recommended on of [20, 40, 80]
lidar_beams = config_file["lidar_beams"]

# The map to use
# Maps can be found in the f1tenth_gym/maps folder,
# and call be dynamically downloaded if they exist in this repo:
# https://github.com/f1tenth/f1tenth_racetracks
map = config_file["map"]

# Reward function to use
# Available reward functions in `git_root`/eml_rl/reward_functions
# Make sure to import requested function
# Example:
# from eml_rl.reward_functions.ProgressReward import ProgressReward
module = importlib.import_module(f"eml_rl.reward_functions.{config_file["reward_function"]}")
reward_function = getattr(module, config_file['reward_function'])

# cl_grid_static: centerline, fixed start point
# cl_random_static: centerline, random
# shuf_random_static: raceline
reset_config = config_file["reset_config"]
