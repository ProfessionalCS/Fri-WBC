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
# import gym??
from PointEnv import PointEnv

print("All imports work!")
print ("Stable_baselines3 Installed Successfully")
print ("Gym Installed Successfully")

def create_point_env():
    target_ee_position = np.array([0.3,0,0])

    env = PointEnv(
        robots="Panda",  
        has_renderer=True,  # Rn I unabled visual sim if its needed
        has_offscreen_renderer=False,
        use_camera_obs=False,
        reward_shaping=True  #  dense reward
    )
    env.set_target_position(target_ee_position)
    # Vectorize and Normalize Environment
    my_wrapped_env = GymWrapper(env)
    my_vec_env = DummyVecEnv([lambda: my_wrapped_env])
    my_vec_env = VecNormalize(my_vec_env, norm_obs=True, norm_reward=True)
    return my_vec_env

def create_lift_env():
    env = suite.make( # robosuite env here
        env_name="Lift", # try with other tasks like "Stack" and "Door"
        robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        reward_shaping = True # dense reward
    )

    # wrapping the environment for being compatible with Gym (complete)
    my_wrapped_env = GymWrapper(env)

    # creating the vector env for stable baselines (I think its needed)
    my_vec_env = DummyVecEnv([lambda: my_wrapped_env])

    # normalizing (scaling the input from 0 to 1) ==> IS IT LIKE Discout?   (complete)
    my_vec_env = VecNormalize(my_vec_env, norm_obs=True, norm_reward=True)



# create environment instance
if __name__ == "__main__":
    env = create_point_env()


    model_name = "point_model"
    file_path = model_name + ".zip"
    if os.path.exists(file_path):
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
