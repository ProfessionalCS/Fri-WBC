import numpy as np
import os
import robosuite as suite
from robosuite.wrappers import GymWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from robosuite.environments.manipulation.lift import Lift

# Custom Lift Environment with Reward Function   ==> I got rid of the cube part since its not needed
class CustomLiftEnv(Lift): # passed an instance of the Lift Env here 
    def reward(self, action):
        # Getting the current position of the gripper
        gripper_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
        # Calculate distance to the target
        distance = np.linalg.norm(gripper_pos - self.current_target_position)
        # Reward based on how close gripper to the target
        reward = 1 - np.tanh(10.0 * distance)
        return reward

    def set_target_position(self, target_position):
        # using this to update the current target position
        self.current_target_position = target_position


if __name__=="__main__":

    target_ee_position = np.array([0.3,0,0])

    env = CustomLiftEnv(
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

    # Set up and train the PPO model
    file_path = "point_model.zip"
    if os.path.exists(file_path):
        print("Loading model")
        model = PPO.load("point_model")
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

    n_epochs = 100
    for i in n _epochs:
        model.learn(
            total_timesteps=25000,  
            log_interval=1
        )

        # Save the trained model
        model.save(f"point_model_{i}")
