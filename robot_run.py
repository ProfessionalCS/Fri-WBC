# Necessary imports https://github.com/Sarcovora/Summer-Camp-Rosbag.git
import numpy as np
import robosuite as suite
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from robosuite.wrappers import GymWrapper
import rospy
from geometry_msgs.msg import Pose
from intera_interface import Limb
import transforms3d as tf3d

# Real Robot Environment Class
class SawyerEnv():
    def __init__(self):
        rospy.init_node('go_to_cartesian_pose_py')
        self.limb = Limb()
        self.tip_name = "right_hand"
        self.rate = rospy.Rate(10)
        self.horizon = 500
        self.step_counter = 0

    def save_pose(self):
        global neturalx, neturaly, neturalz, netural2x, netural2y, netural2z, netural2w
        tempVar = env.limb.endpoint_pose()["position"]
        tempVar2 = env.limb.endpoint_pose()["orientation"]
        neturalx, neturaly, neturalz, netural2x, netural2y, netural2z, netural2w = tempVar.x, tempVar.y, tempVar.z, tempVar2.x, tempVar2.y, tempVar2.z, tempVar2.w

    def reset(self):
        self.step_counter = 0
        self.go_to_cartesian(neturalx, neturaly, neturalz, netural2x, netural2y, netural2z, netural2w)

    def go_to_cartesian(self, x, y, z, qx, qy, qz, qw):
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        pose.orientation.x = qx
        pose.orientation.y = qy
        pose.orientation.z = qz
        pose.orientation.w = qw
        joint_angles = self.limb.ik_request(pose, self.tip_name)
        if joint_angles:
            self.limb.set_joint_positions(joint_angles)
        else:
            rospy.logerr("IK request failed for the given pose.")

    def step(self, action):
        self.step_counter += 1
        current_pose = self.limb.endpoint_pose()
        current_matrix = self._pose_to_matrix(current_pose["position"], current_pose["orientation"])
        action_matrix = self._action_to_matrix(action)
        new_matrix = np.dot(action_matrix, current_matrix)

        # Extract new pose
        new_position = new_matrix[:3, 3]
        new_orientation = tf3d.quaternions.mat2quat(new_matrix[:3, :3])
        self.go_to_cartesian(*new_position, *new_orientation)

        # Return dummy values for now
        reward = 0
        done = self.step_counter >= self.horizon
        obs = {"pose": {"position": new_position, "orientation": new_orientation}}
        return obs, reward, done, {}

    def _pose_to_matrix(self, position, orientation):
        matrix = np.eye(4)
        matrix[:3, :3] = tf3d.quaternions.quat2mat(
            [orientation.w, orientation.x, orientation.y, orientation.z]
        )
        matrix[:3, 3] = [position.x, position.y, position.z]
        return matrix

    def _action_to_matrix(self, action):
        matrix = np.eye(4)
        translation = action[0]
        quaternion = [action[1][3], action[1][0], action[1][1], action[1][2]]
        matrix[:3, :3] = tf3d.quaternions.quat2mat(quaternion)
        matrix[:3, 3] = translation
        return matrix


# Integration with the Simulation Loop
if __name__ == "__main__":
    # this could be complete nonsense I have not tried it and had to abandon due to time constraints 
    # Real Robot Env
    real_env = SawyerEnv()
    real_env.save_pose()

    # Simulated Environment
    env = suite.make(
        env_name="GoToPointTask",
        robots="Sawyer",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        #! Goes to default value if not provided
    )
    gym_env = GymWrapper(env)
    env = DummyVecEnv([lambda: gym_env])
    env = VecNormalize.load("./data_and_models/training_models/vec_normalize.pkl", env)

    # Load the policy
    model = PPO.load("./data_and_models/training_models/point_model.zip", env=env)
    obs = env.reset()

    # Deployment Loop I have no idea if this works
    for i in range(10000):
        # Predict action from the trained model
        action, _states = model.predict(obs)

        # Execute action in the real robot
        real_env.step(action)

        # Render for simulation visualization (optional)
        gym_env.render()

        # Check termination
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()
