import robosuite as suite
import gym
import matplotlib.pyplot as plt
from IPython import display
import numpy as np

# Create the environment
env = suite.make( # robosuite env here
    env_name="Lift", # try with other tasks like "Stack" and "Door"
    robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
)

# Start the environment
state = env.reset()

for i in range(1000):
     action = np.random.randn(env.robots[0].dof) # change to learned policy from st baselines to rl
     obs, reward, done, info = env.step(action)  
     env.render()  

env.close()