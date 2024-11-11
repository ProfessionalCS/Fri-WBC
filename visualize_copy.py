# import cv2
import numpy as np
import robosuite as suite

from stable_baselines3 import PPO     # PPO Model Being trained using  cpu device
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import robosuite as suite
from robosuite.wrappers import GymWrapper

import torch
from stable_baselines3.common.vec_env import VecNormalize
# File Finding 
from pathlib import Path  
from PointEnv import PointEnv
# from robosuite.environments.base import REGISTERED_ENVS  # loads wrong environment!!!
import gym
from stable_baselines3.common.utils import set_random_seed
from robosuite.environments.base import register_env
from my_environments import GoToPointTask


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
def make_gym_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        
        env = gym.make(env_id, reward_type="dense")
        env = gym.wrappers.FlattenObservation(env)
        env = Monitor(env)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

if __name__=="__main__":

    register_env(GoToPointTask)
    env_options = {}
    env_options["control_freq"] = 20
    env_options["render_camera"] = None
    env_options["use_object_obs"] = False
    env_options["horizon"] = 1000

    seed = 3
    env = make_robosuite_env("GoToPointTask",env_options, 1, seed)()

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
        if isinstance(obs, tuple):
            observation_array = obs[0]
        else:
            observation_array = obs
        print("Model observation space shape:", model.observation_space.shape)
        print("Environment observation shape:", observation_array.shape)



        # Check if padding is needed to match the expected observation space (42,)
        if observation_array.shape[0] < 32:
            # Pad with zeros or concatenate additional parts from the observation dict if necessary
            observation_array = np.pad(observation_array, (0, 32 - observation_array.shape[0]), 'constant')
            # np. pad used from the outside source recommendation

        # np.predict used to predict the observation array value 
        action, _states = model.predict(observation_array)
        print(f"Action taken: {action}")

        obs, reward, done, info = env.step([action])
        print(f"Reward: {reward}, Done: {done}, Info: {info}")
        env.render()

        if done:
            obs = env.reset()

    env.reset()

