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

# from robosuite.environments.base import REGISTERED_ENVS  # loads wrong environment!!!


print("All imports work!")
print ("Gym Installed Successfully")


# create environment instance
env = suite.make( # robosuite env here
    env_name="Lift", # try with other tasks like "Stack" and "Door"
    robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    reward_shaping=True
)

# wrapping the environment for being compatible with Gym (complete)
my_wrapped_env = GymWrapper(env)

# creating the vector env for stable baselines (I think its needed)
my_vec_env = DummyVecEnv([lambda: my_wrapped_env])

# normalizing (scaling the input from 0 to 1) ==> IS IT LIKE Discout?   (complete)
my_vec_env = VecNormalize(my_vec_env, norm_obs=True, norm_reward=True)


# if I want to keep training I can load it from where I left off (this will load model from zip)
model = PPO.load("model_saved_must_work.zip", env=my_vec_env)

######################################################################

# Attempting to Load the Environmnet (Successfuly Completed)

obs = env.reset()
print("Initial observation:", obs)
print("Expected observation space:", my_vec_env.observation_space)

for i in range(1000):
    # Extract the 'robot0_proprio-state' as the base observation
    observation_array = obs['robot0_proprio-state']

    # Check if padding is needed to match the expected observation space (42,)
    if observation_array.shape[0] < 42:
        # Pad with zeros or concatenate additional parts from the observation dict if necessary
        observation_array = np.pad(observation_array, (0, 42 - observation_array.shape[0]), 'constant')
        # np. pad used from the outside source recommendation

    # np.predict used to predict the observation array value 
    action, _states = model.predict(observation_array)
    print(f"Action taken: {action}")

    obs, reward, done, info = env.step(action)
    print(f"Reward: {reward}, Done: {done}, Info: {info}")
    env.render()

    if done:
        obs = env.reset()

env.reset()

