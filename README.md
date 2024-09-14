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
   sudo apt install libgl1-mesa-glx
   ```
6. Install local packages
     ```
     pip install -e f1tenth_gym
     pip install -e f1tenth_planning
     pip install -e stable_baselines3
     pip install -e rl-baselines3-zoo
     pip install -e .
     ```
# Train
```
python3 eml_rl/train.py
```

# Evaluate
```
python3 eml_rl/eval.py <path to model zip>
```

# Hyperparameter Tuning
```
./eml_rl/opt.sh
```
