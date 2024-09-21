# Installation
1. Clone the repo
   ```
   git clone https://github.com/r-clifford/EML-RL.git --recursive && cd EML-RL
   ```
3. Create Python Virtual Environment
   ```
   python3 -m venv rl_venv
   ```
   > If this outputs a warning, follow the instructions to install the venv package. Re-run the above command
   ```
   The virtual environment was not created successfully because ensurepip is not available.  On Debian/Ubuntu systems, you need to install the python3-venv package using the following command.

     apt install python3.10-venv

   You may need to use sudo with that command.  After installing the python3-venv package, recreate your virtual environment.
   ```

5. Activate the virtual environment
   ```
   source rl_venv/bin/activate
   ```

6. Install dependencies
   ```
   sudo apt update
   # This may fail on certain distro versions
   # Try to continue with out it
   sudo apt install libgl1-mesa-glx
   ```
6. Install local packages
     ```
     pip install -e f1tenth_gym
     pip install -e f1tenth_planning
     pip install -e stable-baselines3
     pip install -e rl-baselines3-zoo
     pip install -e stable-baselines3-contrib
     pip install -e .
     pip install -r requirements.txt
    ```
# Train
https://stable-baselines3.readthedocs.io/en/master/guide/quickstart.html
```
python3 eml_rl/train.py <algorithm>
```
## Train with rl-baselines3-zoo
https://rl-baselines3-zoo.readthedocs.io/en/master/guide/quickstart.html
```
./eml_rl/train.sh <log_dir> <algorithm> <config_file>
```
- `log_dir`: directory for log storage
- `algorithm`: RL algorithm to use, see notes
- `config_file`: Config file with environment parameters and model hyperparams
   - See `eml_rl/config/hyperparams/<algo>_f1tenth.py`
# Evaluate
```
python3 eml_rl/eval.py <algorithm> <path to model zip>
```

# Hyperparameter Tuning
https://rl-baselines3-zoo.readthedocs.io/en/master/guide/tuning.html
```
./eml_rl/opt.sh <log_dir> <algorithm>
```
- See section on training with rl-baselines3-zoo
# Notes
- Pull latest changes with `git pull --recurse-submodules`
- Remember to source the virtual environment with `source rl_venv/bin/activate`
- Current valid algorithms
  - ppo
  - sac
  - td3
  - tqc
  - rppo is not currently working
  > td3, sac, and tqc have had the best results
- The reward function is defined in `eml_rl/reward.py` and subclasses the `Reward` class defined by `f1tenth_gym`
- Action and observation space transforms are defined in `eml_rl/f1tenth_transforms.py`
- A function to make the environment is defined in `eml_rl/utils.py`
- Domain randomization support is also in `eml_rl/utils.py`
