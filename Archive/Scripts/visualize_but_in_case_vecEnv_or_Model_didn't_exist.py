import numpy as np
import robosuite as suite

from stable_baselines3 import PPO  # PPO Model Being trained using  cpu device
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from robosuite.wrappers import GymWrapper

import torch
from pathlib import Path  
from PointEnv import PointEnv
import gym
from stable_baselines3.common.utils import set_random_seed
from my_environments import GoToPointTask  # Assuming go_to_point.py is in a directory named `my_environments`

print("All imports work!")
print( "Gym Installed Successfully")

def make_robosuite_env(env_id, options, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param options: (dict) additional argumen,mASDts to pass to the specific environment class initializer
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        temp_env = suite.make(
            env_name="GoToPointTask",  # try with other tasks like "Stack" and "Door"
            robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
            has_renderer=True,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            **options
        )
        env = GymWrapper(temp_env)
        env = Monitor(env)
        return env
    return _init

def make_gym_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        
        env = gym.make(env_id, reward_type="dense")
        env = Monitor(env)
        env = gym.wrappers.FlattenObservation(env)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

if __name__ == "__main__":

    # commented out because it gave me error
    # register_env(GoToPointTask)
    # env_id = "GoToPointTask"

    # Define environment options
    env_options = {}
    env_options["control_freq"] = 20
    env_options["render_camera"] = None
    env_options["use_object_obs"] = False
    env_options["horizon"] = 1000
    options = env_options

    # Initialize the GoToPointTask environment with robosuite
    env = suite.make(
        env_name="GoToPointTask",  # try with other tasks like "Stack" and "Door"
        robots="Panda",   # try with other robots like "Sawyer" and "Jaco"
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        **options
    )

    # Wrap with GymWrapper and DummyVecEnv for compatibility with Stable-Baselines3
    seed = 3
    env_gym = GymWrapper(env)
    env = DummyVecEnv([lambda: env_gym])

##############################################################################################################
    # This can help if problems with loading 
    vec_normalize_path = Path("./training_models/vec_normalize.pkl") # path to Vec Normalize as on lab machine
    
    if vec_normalize_path.is_file() and vec_normalize_path.stat().st_size > 0: # checking if file exists simple stuff
        try:
            env = VecNormalize.load(vec_normalize_path, env) # load this vecNormalize
            print("VecNormalize loaded successfully.")   # printing just to know if it worked
        except EOFError: # if file not working 
            print("VecNormalize doesn't exist. Create new VecNormalize.")
            env = VecNormalize(env) # just not working
    else:
        print("VecNormalize not found. Creating new VecNormalize.")
        env = VecNormalize(env)

    # Check if the model file exists and is valid
    model_path = Path("./training_models/point_model.zip") # just put path and it will be set

    if model_path.is_file() and model_path.stat().st_size > 0: # checking if it exists
        try:
            model = PPO.load(model_path, env=env)
            print("Model loaded successfully.") # success
        except ValueError:
            print("Model doesn't exist. Creating a new model.")
            model = PPO("MlpPolicy", env, verbose=1) # not success
    else:
        print("Model not found. Creating a new model.")
        model = PPO("MlpPolicy", env, verbose=1) # need to create model if no model found

##############################################################################################################
    
    # Reset the environment and begin interaction
    obs = env.reset()
    print("Initial observation:", obs)

    # run model this number of steps if needed
    for i in range(100000):
        action, _states = model.predict(obs)

        obs, reward, done, info = env.step(action)

        # I think this helps to keep track of data
        print(f"Reward: {reward}, Done: {done}, Info: {info}") # Adjust this based on what step() returns

        env_gym.render()

        if done:
            obs = env.reset() # Reset environment if done
