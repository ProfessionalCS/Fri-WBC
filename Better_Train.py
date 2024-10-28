# import cv2
import numpy as np
import robosuite as suite

from stable_baselines3 import PPO     # PPO Model Being trained using  cpu device
from stable_baselines3.common.vec_env import DummyVecEnv
import robosuite as suite
from robosuite.wrappers import GymWrapper

import torch
from stable_baselines3.common.vec_env import VecNormalize
# File Finding 
from pathlib import Path  
import os

from PointEnv import PointEnv

# Register Enviorment
#import Gymnasium
from robosuite.environments.base import register_env
from my_environments import GoToPointTask
register_env(GoToPointTask)

print("All imports work!")
print ("Stable_baselines3 Installed Successfully")
print ("Gym Installed Successfully")




# create environment instance
if __name__ == "__main__":
    env = suite.make(
    env_name="GoToPointTask", # try with other tasks like "Stack" and "Door"
    robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
)



    model_name = "point_model"
    file_path = model_name + ".zip"
    if os.path.exists(file_path) & False:
        print("Loading model")
        model = PPO.load(model_name)
        model.env = env

    else : 
        # Initialize PPO model with parameters included to facilitate more effcient training (parameters included as per request) (complete)
        model = PPO(
            policy= "MlpPolicy",      # Policy that exists
            env= env,          # Normalized vectorized env
            learning_rate= 0.0003,    # Learning rate for the model (should it be changed?)
            n_steps= 3000,            # Number of steps for each update
            batch_size= 500,           # Batch size for optimization
            verbose=1,                # Verbose idk what that is
            tensorboard_log='/home/fri/tb.log'
        )

    # making the model learn (train)  (complete)
    model.learn(
        total_timesteps= 25000,  # Number of timesteps for model training
        log_interval= 1,        # Interval for training progress,
    )

    # saving the model in general
    model.save(model_name)
