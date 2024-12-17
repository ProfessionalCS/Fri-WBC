from robosuite.environments.manipulation.lift import Lift
import numpy as np
import os
import robosuite as suite


class PointEnv(Lift): # passed an instance of the Lift Env here 
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