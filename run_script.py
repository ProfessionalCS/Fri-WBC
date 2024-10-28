print("This was called")
import cv2
import numpy as np
import robosuite as suite

from stable_baselines3 import PPO     # PPO Model Being trained using  cpu device
from stable_baselines3.common.vec_env import DummyVecEnv
import robosuite as suite
from robosuite.wrappers import GymWrapper

import torch
from stable_baselines3.common.vec_env import VecNormalize

# from robosuite.environments.base import REGISTERED_ENVS  # loads wrong environment!!!


print("All imports work!")
print ("Stable_baselines3 Installed Successfully")
print ("Gym Installed Successfully")

# create environment instance
env = suite.make( # robosuite env here
    env_name="Lift", # try with other tasks like "Stack" and "Door"
    robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
)

# wrapping the environment for being compatible with Gym (complete)
my_wrapped_env = GymWrapper(env)

# creating the vector env for stable baselines (I think its needed)
my_vec_env = DummyVecEnv([lambda: my_wrapped_env])

# normalizing (scaling the input from 0 to 1) ==> IS IT LIKE Discout?   (complete)
my_vec_env = VecNormalize(my_vec_env, norm_obs=True, norm_reward=True)

# Initialize PPO model with parameters included to facilitate more effcient training (parameters included as per request) (complete)
modelInitialize = PPO(
    policy= "MlpPolicy",      # Policy that exists
    env= my_vec_env,          # Normalized vectorized env
    learning_rate= 0.0003,    # Learning rate for the model (should it be changed?)
    n_steps= 3000,            # Number of steps for each update
    batch_size= 500,           # Batch size for optimization
    verbose=1,                # Verbose idk what that is
    tensorboard_log='/home/anastasiia/tb.log/'
)

# making the model learn (train)  (complete)
modelInitialize.learn(
    total_timesteps= 50000,  # Number of timesteps for model training
    log_interval= 1,        # Interval for training progress,

)

# saving the model in general
modelInitialize.save("model_saved_must_work")    # is this needed?   // how to add joints? and make it model.learn.joints(actions) not states  its actions

# saving model in pth format as was mentioned in instructions
torch.save(modelInitialize.policy.state_dict(), "model_saved_must_work.pth")   # or this is better?
breakpoint()
# loading the model
model = PPO.load("model_saved_must_work.zip")

# if I want to keep training I can load it from where I left off (this will oad model from zip)
model = PPO.load("model_saved_must_work.zip", env=my_vec_env)

######################################################################

# Attempting to Load the Environmnet (Successfuly Completed)

obs = env.reset()
print("Initial observation:", obs)
print("Expected observation space:", my_vec_env.observation_space)

# Visualize the ENV   Can make a video saving each timr
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

# #creating a trjectory ==> desired joint angles and end effector position
# trajectoryMovementHabd = {
#     [10, 10, 10, 10]
#     [20, 20, 20, 20]
#     [30, 30, 30, 30]
# }

# gripper ==
#matching trajectiry of the hand matching position of te joints

####################################################################
# for i in range(1000):  # Run for 1000 steps
#     action, _states = model.predict(obs)  
#     obs, reward, done, info = env.step(action)     
#     env.render() 
    
#     if done:  # If the episode is done, reset the environment
#         obs = env.reset()

# reset the environment

env.reset()




# Other ATTEMPTS

# SAME THINGS AS ABOVE FOR MODEL LEARNING)
# make the model learn  over 10000 time stemps (are they same as epochs?)
# modelInitialize.learn(total_timesteps=10000)



# initialize stable baselines model with this env
# modelInitialize = PPO("MlpPolicy", my_vec_env, verbose=1) (SAME THINGS AS BELOW BUT WITH LESS PARAMS)

# use lambda function to shorten the code
# def make_env():
#     return my_wrapped_env

# my_vec_env = DummyVecEnv([make_env]) <==== could use this but it will be longer code both parts should replace "my_vec_env = DummyVecEnv([lambda: my_wrapped_env])"




# Instructions
# created robusiste env == > lift use 
# call of this in this file??? ===> look rl example with stable baselines on youtube

# 
# wrapp it for gym
# initilize stable baselines model with this env
# env.learn() ==> save it as pth ==> save
# 

# create the env obosuite env pass gymwrapper to get env (all defined in lift) ==. make env that compatible==> gymWrapper takes in robosuite env and gives compatibility
# stable baselines model ==> as a parameter it takes this env ==> use stable baselines
# env.learn() ==> include parameters also they have normalization  ==> train it
# 


# Load the normalized environment
# Loadthe model ==> rl model trained ==> how, and what model? how to se up the model?
# 

# saved training model inclide
# 
# 
# env = 


# for i in range(1000):
#     action = np.random.randn(env.robots[0].dof) # change to learned policy from st baselines to rl
#     obs, reward, done, info = env.step(action)  
#     env.render()  


    # learn the policy