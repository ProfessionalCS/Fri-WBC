import numpy as np
import os
import robosuite as suite
from robosuite.wrappers import GymWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from robosuite.environments.manipulation.lift import Lift

# Defining Trajectory Waypoints for the gripper ==> each movement in 3D space defined by x, y, z
trajectoryGripper = [
    np.array([0.1, 0.0, 0.0]),  # for x
    np.array([0.0, 0.1, 0.0]),  # for y
    np.array([0.0, 0.0, 0.1])   # for z   ==> self-note to add commas to avoid errors
]

# Custom Lift Environment with Custom Reward Function
class CustomLiftEnv(Lift):
    def reward(self, action=None):  # Override the existing reward method
        reward = 0.0
        # Getting the current position of the gripper
        gripper_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
        target_position_needed = np.array([0.0, 0.2, 0.0])  # Update if ==>  needed
        # Distance between the gripper and the target
        distance = np.linalg.norm(gripper_pos - target_position_needed)
        # Setting the reward 
        reward_giving = 1 - np.tanh(10.0 * distance)
        reward += reward_giving  # This will increase the reward each time
        return reward

# Set up the custom environment
env = CustomLiftEnv(  # Create an instance of CustomLiftEnv to ensure custom reward is used
    robots="Panda",  
    has_renderer=False,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    reward_shaping=True  # Enable dense reward
)

# Wrap and Vectorize the Environment
my_wrapped_env = GymWrapper(env)  # Wrapped environment
my_vec_env = DummyVecEnv([lambda: my_wrapped_env])
my_vec_env = VecNormalize(my_vec_env, norm_obs=True, norm_reward=True)  # Normalized

# Set up the PPO Model
file_path = "model_saved_must_work.zip"
if os.path.exists(file_path):
    print("Loading model")
    model = PPO.load("model_saved_must_work")
    model.env = my_vec_env
else:
    model = PPO(
        policy="MlpPolicy",      
        env=my_vec_env,          
        learning_rate=0.0003,   
        n_steps=3000,            
        batch_size=500,          
        verbose=1,                
        tensorboard_log='/home/fri/tb.log' 
    )

# Train the model
model.learn(
    total_timesteps=2500000,  # Determine when model is most performing based on the reward  
    log_interval=1  # Interval for training progress
)

# Save the trained model
model.save("model_saved_must_work")