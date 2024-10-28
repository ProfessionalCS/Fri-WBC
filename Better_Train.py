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

from PointEnv import PointEnv

# Register Enviorment
#import Gymnasium
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



def make_robosuite_env(env_id, options, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param options: (dict) additional arguments to pass to the specific environment class initializer
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = GymWrapper(suite.make(env_id, **options))
        env = Monitor(env)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
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


# create environment instance
if __name__ == "__main__":
    register_env(GoToPointTask)
    with open("rl_config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)

    env_options = config["robosuite"]

    env_id = env_options.pop("env_id")

    # Settings for stable-baselines RL algorithm
    sb_config = config["sb_config"]
    training_timesteps = sb_config["total_timesteps"]
    check_pt_interval = sb_config["check_pt_interval"]
    num_cpu = sb_config["num_cpu"]

    # Settings for stable-baselines policy
    policy_kwargs = config["sb_policy"]
    policy_type = policy_kwargs.pop("type")

    # Settings used for file handling and logging (save/load destination etc)
    file_handling = config["file_handling"]

    tb_log_folder = file_handling["tb_log_folder"]
    tb_log_name = file_handling["tb_log_name"]

    save_model_folder = file_handling["save_model_folder"]
    save_model_filename = file_handling["save_model_filename"]
    load_model_folder = file_handling["load_model_folder"]
    load_model_filename = file_handling["load_model_filename"]

    continue_training_model_folder = file_handling["continue_training_model_folder"]
    continue_training_model_filename = file_handling["continue_training_model_filename"]

    # Join paths
    save_model_path = os.path.join(save_model_folder, save_model_filename)
    save_vecnormalize_path = os.path.join(save_model_folder, 'vec_normalize_' + save_model_filename + '.pkl')
    load_model_path = os.path.join(load_model_folder, load_model_filename)
    load_vecnormalize_path = os.path.join(load_model_folder, 'vec_normalize_' + load_model_filename + '.pkl')

    # Settings for pipeline
    training = config["training"]
    seed = config["seed"]

    # RL pipeline
    if training:
        env = SubprocVecEnv([make_robosuite_env(env_id, env_options, i, seed) for i in range(num_cpu)]) 

        # Create callback
        checkpoint_callback = CheckpointCallback(save_freq=check_pt_interval, save_path='./checkpoints/', 
                                name_prefix=save_model_filename, verbose=2)
        
        # Train new model
        if continue_training_model_filename is None:

            # Normalize environment
            env = VecNormalize(env)

            # Create model
            model = PPO(policy_type, env, policy_kwargs=policy_kwargs, tensorboard_log=tb_log_folder, verbose=1)

            print("Created a new model")

        # Continual training
        else:

            # Join file paths
            continue_training_model_path = os.path.join(continue_training_model_folder, continue_training_model_filename)
            continue_training_vecnormalize_path = os.path.join(continue_training_model_folder, 'vec_normalize_' + continue_training_model_filename + '.pkl')

            print(f"Continual training on model located at {continue_training_model_path}")

            # Load normalized env 
            env = VecNormalize.load(continue_training_vecnormalize_path, env)

            # Load model
            model = PPO.load(continue_training_model_path, env=env)

        # Training
        model.learn(total_timesteps=training_timesteps, tb_log_name=tb_log_name, callback=checkpoint_callback, reset_num_timesteps=True)

        # Save trained model
        model.save(save_model_path)
        env.save(save_vecnormalize_path)

    else:
        # Create evaluation environment
        env_options['has_renderer'] = True
        env_gym = GymWrapper(suite.make(env_id, **env_options)) 
        env = DummyVecEnv([lambda : env_gym])

        # Load normalized env
        env = VecNormalize.load(load_vecnormalize_path, env)

        # Turn of updates and reward normalization
        env.training = False
        env.norm_reward = False

        # Load model
        model = PPO.load(load_model_path, env)

        # Simulate environment
        obs = env.reset()
        eprew = 0
        while True:
            action, _states = model.predict(obs)
            print(f"action: {action}")
            obs, reward, done, info = env.step(action)
            #print(action)
            print(f'reward: {reward}')
            eprew += reward
            env_gym.render()
            if done:
                print(f'eprew: {eprew}')
                obs = env.reset()
                eprew = 0

        env.close()



    
    # env = SubprocVecEnv([make_robosuite_env(env_id, env_options, i, seed) for i in range(5)]) # Hard coded cpu count


    # model_name = "point_model"
    # file_path = model_name + ".zip"
    # if os.path.exists(file_path) & False:
    #     print("Loading model")
    #     model = PPO.load(model_name)
    #     model.env = env

    # else : 
    #     # Initialize PPO model with parameters included to facilitate more effcient training (parameters included as per request) (complete)
    #     model = PPO(
    #         policy= "MlpPolicy",      # Policy that exists
    #         env= env,          # Normalized vectorized env
    #         learning_rate= 0.0003,    # Learning rate for the model (should it be changed?)
    #         n_steps= 3000,            # Number of steps for each update
    #         batch_size= 500,           # Batch size for optimization
    #         verbose=1,                # Verbose idk what that is
    #         tensorboard_log='/home/fri/tb.log'
    #     )

    # # making the model learn (train)  (complete)
    # model.learn(
    #     total_timesteps= 25000,  # Number of timesteps for model training
    #     log_interval= 1,        # Interval for training progress,
    # )

    # saving the model in general
    model.save(model_name)
