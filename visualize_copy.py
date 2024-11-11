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
from PointEnv import PointEnv
# from robosuite.environments.base import REGISTERED_ENVS  # loads wrong environment!!!


print("All imports work!")
print ("Gym Installed Successfully")


def make_robosuite_env(env_id, options, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param options: (dict) additional arguments to pass to the specific environment class initializer
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    
    def _init():
        
        temp_env = suite.make(
        env_name="GoToPointTask", # try with other tasks like "Stack" and "Door"
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

def create_lift_env():
    env = suite.make( # robosuite env here
        env_name="Lift", # try with other tasks like "Stack" and "Door"
        robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
        has_renderer=True,
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
    return my_vec_env

if __name__=="__main__":
    env = create_point_env()

    model_name = "point_model"
    model_path = model_name + ".zip"
    model = PPO.load(model_path, env=env)


    obs = env.reset()
    print("Initial observation:", obs)
    # print("Expected observation space:", my_vec_env.observation_space)
    
    for i in range(1000):
        # Extract the 'robot0_proprio-state' as the base observation
        # breakpoint()
        # observation_array = obs['robot0_proprio-state']
        observation_array = obs[0]

        # Check if padding is needed to match the expected observation space (42,)
        if observation_array.shape[0] < 42:
            # Pad with zeros or concatenate additional parts from the observation dict if necessary
            observation_array = np.pad(observation_array, (0, 42 - observation_array.shape[0]), 'constant')
            # np. pad used from the outside source recommendation

        # np.predict used to predict the observation array value 
        action, _states = model.predict(observation_array)
        # print(f"Action taken: {action}")

        obs, reward, done, info = env.step([action])
        # print(f"Reward: {reward}, Done: {done}, Info: {info}")
        env.render()

        if done:
            obs = env.reset()

    env.reset()

