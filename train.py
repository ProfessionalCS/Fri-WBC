# import cv2
import numpy as np
import robosuite as suite
import yaml 
from stable_baselines3 import PPO     # PPO Model Being trained using  cpu device
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
import robosuite as suite
from robosuite.wrappers import GymWrapper

import torch
from stable_baselines3.common.vec_env import VecNormalize
# File Finding 
from pathlib import Path  
import os

from archive.PointEnv import PointEnv

# Register Enviorment
import gym
from stable_baselines3.common.utils import set_random_seed
from robosuite.environments.base import register_env
from my_environments import GoToPointTask




from stable_baselines3.common.save_util import save_to_zip_file, load_from_zip_file
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback


print("All imports work!")
print ("Stable_baselines3 Installed Successfully")
print ("Gym Installed Successfully")
print("this is running Better_Train.py")



def make_robosuite_env(env_id, options, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param options: (dict) additional arguments to pass to the specific environment class initializer
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    
    def _init():     
        if rank % 4 == 0:  # first one testing
            target_coordinate_temp = np.array([0.1, 0.1, 0.1])
        elif rank % 4 == 1:  # second one testing
            target_coordinate_temp = np.array([0.2, 0.2, 0.2])
        elif rank % 4 == 2:  # third one testing
            target_coordinate_temp = np.array([0.3, 0.3, 0.3])
        elif rank % 4 == 3:  # fourth one testing
            target_coordinate_temp = np.array([0.4, 0.4, 0.4])
            
        
        temp_env = suite.make(
        env_name="GoToPointTask", # try with other tasks like "Stack" and "Door"
        robots="Sawyer",  # try with other robots like "Sawyer" and "Jaco"
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        target_coordinate = target_coordinate_temp,
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
        env = gym.wrappers(env)
        env = gym.wrappers.FlattenObservation(env)
        env = Monitor(env)
        
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init


# create environment instance
if __name__ == "__main__":
    
    register_env(GoToPointTask)
    env_options = {}
    env_options["control_freq"] = 20
    env_options["render_camera"] = None
    env_options["use_object_obs"] = False
    env_options["horizon"] = 1000
    
     # Define the target coordinate
    # target_coordinate = np.array([0.5, 0.5, 0.5])

    # Add the target coordinate to the environment options
    # env_options["target_coordinate"] = target_coordinate # Default target coordinate is [0.5, 0.5, 0.5]
    
    
    seed = 3
    num_cpu = 8
    env = SubprocVecEnv([make_robosuite_env("GoToPointTask",env_options, i, seed) for i in range(num_cpu)])# Hard coded cpu count

    model_name = "./data_and_models/training_models/point_model"
    vec_path = "./data_and_models/training_models/vec_normalize.pkl"
    file_path = model_name + ".zip"
    
    if os.path.exists(file_path) :
        print("Loading model")
        env = VecNormalize.load(vec_path, env)
        model = PPO.load(model_name)
        model.env = env

    else : 
        # Initialize PPO model with parameters included to facilitate more effcient training (parameters included as per request) (complete)
        print("Creating new model")
        env = VecNormalize(env)
        model = PPO(
            policy= "MlpPolicy",      # Policy that exists
            env= env,          # Normalized vectorized env
            learning_rate= 0.0003,    # Learning rate for the model (should it be changed?)
            n_steps= 3000,            # Number of steps for each update
            batch_size= 500,           # Batch size for optimization
            verbose=1,                # Verbose idk what that is
            tensorboard_log='./data_and_models/log/tb.log'
        )
    checkpoint_callback = CheckpointCallback(save_freq=5000, save_path='./data_and_models/checkpoints/', name_prefix='rl_model')


    # making the model learn (train)  (complete)
    model.learn(
        total_timesteps= 100000,  # Number of timesteps for model training
        log_interval= 1,        # Interval for training progress,
        callback=checkpoint_callback
    )
    model.save(model_name)
    env.save(vec_path)
