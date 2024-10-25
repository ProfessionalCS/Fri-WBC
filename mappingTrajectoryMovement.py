import numpy as np
import os
import robosuite as suite
from robosuite.wrappers import GymWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from robosuite.environments.manipulation.lift import Lift

# Define the action creation function
def create_gripper_action(target_position, target_orientation, gripper_control):
    action = np.concatenate((target_position, target_orientation, [gripper_control]))
    return action

# Define movement increment (10 cm or 0.1 meters)
movement_step = 0.1  # 10 cm per step
total_steps = 10  # Move 10 times by 10 cm each

# Function to generate the gripper's trajectory
def generate_gripper_trajectory():
    trajectory = []
    current_position = np.array([0.0, 0.0, 0.0])  # Start position
    for _ in range(total_steps):
        current_position += np.array([movement_step, 0.0, 0.0])  # Move 10 cm along the x-axis
        trajectory.append(np.copy(current_position))  # Store new position
    return trajectory

# Initialize the trajectory
trajectory = generate_gripper_trajectory()

# Define target orientation 
target_gripper_orientation = np.array([0, 0, 0, 1])

# Custom Lift Environment with Reward Function   ==> I got rid of the cube part since its not needed
class CustomLiftEnv(Lift): # passed an instance of the Lift Env here 
    def reward(self, action):
        reward = 0.0
        # Getting the current position of the gripper
        gripper_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
        # Calculate distance to the target
        distance = np.linalg.norm(gripper_pos - self.current_target_position)
        # Reward based on how close gripper to the target
        reward_giving = 1 - np.tanh(10.0 * distance)
        reward = reward + reward_giving
        return reward

    def set_target_position(self, target_position):
        # using this to update the current target position
        self.current_target_position = target_position

# Cretaed an environment ==> 
env = CustomLiftEnv(
    robots="Panda",  
    has_renderer=True,  # Rn I unabled visual sim if its needed
    has_offscreen_renderer=False,
    use_camera_obs=False,
    reward_shaping=True  #  dense reward
)

# Using the trajectory to guide gripper movements
env.reset()
for target_position in trajectory:
    env.set_target_position(target_position)
    gripper_action = create_gripper_action(target_position, target_gripper_orientation, 0)  # Corrected call
    obs, reward, done, info = env.step(gripper_action)
    print("Gripper Moving to Position:", env.sim.data.site_xpos[env.robots[0].eef_site_id])

 # Ensuring that gripper actually moves

#############################################@@@ Model Initialized 
# Vectorize and Normalize Environment
my_wrapped_env = GymWrapper(env)
my_vec_env = DummyVecEnv([lambda: my_wrapped_env])
my_vec_env = VecNormalize(my_vec_env, norm_obs=True, norm_reward=True)

# Set up and train the PPO model
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

model.learn(
    total_timesteps=2500000,  
    log_interval=1
)

# Save the trained model
model.save("model_saved_must_work")

################################################################


# # # Defining Trajectory Waypoints for the gripper ==> each movement in 3D space defined by x, y, z
# # trajectoryGripper = [
# #     np.array([0.1, 0.0, 0.0]),  # for x
# #     np.array([0.0, 0.1, 0.0]),  # for y
# #     np.array([0.0, 0.0, 0.1])   # for z   ==> self-note to add commas to avoid errors
# # ]

# # target_gripper_position = np.array([0.2, 0.1, 0.3])  
# # target_gripper_orientation = np.array([0, 0, 0, 1])

# movement_step = 0.1 # 10 cm
# total_steps = 10  # 10 steps for 10 cm
# # initialize trajectory
# trajectory = generate_gripper_trajectory()

# target_gripper_orientation = np.array([0, 0, 0, 1])

# class CustomLiftEnv(Lift):
#     def reward(self):  # Removed unused 'action' parameter
#         reward = 0.0
#         # Getting the current position of the gripper
#         gripper_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
#         # Calculate distance to the nearest point on the trajectory
#         distance = np.linalg.norm(gripper_pos - self.current_target_position)
#         # Reward based on proximity to the target
#         reward_giving = 1 - np.tanh(10.0 * distance)
#         reward += reward_giving
#         return reward

#     def set_target_position(self, target_position):
#         # Method to update the current target position
#         self.current_target_position = target_position

        
# def generate_gripper_trajectory():
#     trajectory = []
#     current_position = np.array([0.0, 0.0, 0.0])


# for _ in range(total_steps):
#         current_position += np.array([movement_step, 0.0, 0.0])  # Move 10 cm along the x-axis
#         trajectory.append(np.copy(current_position))  # Store  new position

#     return trajectory



# def create_gripper_action(target_position, target_orientation):
#     # Combine position and orientation into a single action array
#     action = np.concatenate((target_position, target_orientation))
#     return action



# # Need to change reward function   ==> I got rid of the cube part since its not needed
# class CustomLiftEnv(Lift):
#     def reward(self, action=None):  # Override the existing reward method
#         reward = 0.0
#         # Getting the current position of the gripper
#         gripper_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
#         print("Gripper Position:", gripper_pos) 
#         target_position_needed = np.array([0.0, 0.2, 0.0])  # Update as needed
#         # Distance between the gripper and the target
#         distance = np.linalg.norm(gripper_pos - target_position_needed)
#         # Setting the reward 
#         reward_giving = 1 - np.tanh(10.0 * distance)
#         reward += reward_giving  # This will increase the reward each time
#         return reward



# # Set up the environment
# env = CustomLiftEnv(
#     robots="Panda",  
#     has_renderer=True,
#     has_offscreen_renderer=False,
#     use_camera_obs=False,
#     reward_shaping=True  # Enable dense reward
# )




# # Also Vectorize and normlize it remember !!!!
# my_wrapped_env = GymWrapper(env) # warpped

# my_vec_env = DummyVecEnv([lambda: my_wrapped_env])

# my_vec_env = VecNormalize(my_vec_env, norm_obs=True, norm_reward=True)  # normalized


# # Set up the PPO Model
# file_path = "model_saved_must_work.zip" 
# if os.path.exists(file_path):
#     print("Loading model")
#     model = PPO.load("model_saved_must_work")
#     model.env = my_vec_env

# else : 
#     model = PPO(
#         policy= "MlpPolicy",      
#         env= my_vec_env,          
#         learning_rate= 0.0003,   
#         n_steps= 3000,            
#         batch_size= 500,          
#         verbose=1,                
#         tensorboard_log='/home/fri/tb.log' 
#     )

# model.learn(
#     total_timesteps= 2500000,  # TO DO:   ==> determine when model is most performing based on the reward  
#     log_interval= 1,        # Interval for training progress, number of timesteps for model training 
# )

# # saving the model in general
# model.save("model_saved_must_work")    
