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
# from pathlib import Path  
# from archive.PointEnv import PointEnv
# from robosuite.environments.base import REGISTERED_ENVS  # loads wrong environment!!!
import gym
from stable_baselines3.common.utils import set_random_seed
from robosuite.environments.base import register_env
from my_environments import GoToPointTask
import random



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
        robots="Sawyer",  # try with other robots like "Sawyer" and "Jaco"
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
    register_env(GoToPointTask)
    env_id = "GoToPointTask"
    env_options = {}
    env_options["control_freq"] = 20
    env_options["render_camera"] = None
    env_options["use_object_obs"] = False
    env_options["horizon"] = 1000
    options = env_options
    
    # Grab random seed 
    seed = 42  # You can replace 42 with any integer value you prefer
    seed = random.randint(0, 4)
    if seed % 4 == 0:  # first one testing
            target_coordinate_temp = np.array([0.5, 0.1, 0.1])
    elif seed % 4 == 1:  # second one testing
            target_coordinate_temp = np.array([0.2, 0.2, 0.2])
    elif seed % 4 == 2:  # third one testing
            target_coordinate_temp = np.array([0.3, 0.3, 0.3])
    elif seed % 4 == 3:  # fourth one testing
            target_coordinate_temp = np.array([0.4, 0.4, 0.4])
    print(f"Seed: {seed} and Target Coordinate: {target_coordinate_temp}")
    
    env =  suite.make(
        env_name="GoToPointTask", # try with other tasks like "Stack" and "Door"
        robots="Sawyer",  # try with other robots like "Sawyer" and "Jaco"
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        target_coordinate = target_coordinate_temp,
        **options
        )
    env_gym = GymWrapper(env)
    env = DummyVecEnv([lambda : env_gym])
    # Load the model
    env = VecNormalize.load("./data_and_models/training_models/vec_normalize.pkl", env)
    model_name = "point_model"
    model_path = "./data_and_models/training_models/point_model.zip"
    model = PPO.load(model_path, env=env)
    obs = env.reset()
    print("Initial observation:", obs)

    for i in range(10000):

        # Predict the action
        action, _states = model.predict(obs)
        #print(f"Action taken: {action}")

        # Pass the action to the environment
        # action = np.array(action)  # Ensure the action is in the correct format

        # Check the return value of step(action)
        # result = env.step(action)
        # print("Step result:", result)
        
        # Now unpack based on the result
        #print("Step result:" + action)
        obs, reward, done, info = env.step(action)  # Adjust this based on what step() returns
        #print(f"Reward: {reward}")#, Done: {done}, Info: {info}")
        print("Action:", action)
        env_gym.render()

        if done:
            obs = env.reset()  # Reset environment if done
