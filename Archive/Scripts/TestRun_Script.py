import numpy as np
import robosuite as suite
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from robosuite.wrappers import GymWrapper
from stable_baselines3.common.vec_env import VecNormalize

# Step 1: Create the environment
try:
    print("Attempting to initialize the environment...")
    env = suite.make(
        env_name="Lift",
        robots="Panda",
        has_renderer=True,           # Ensure the renderer is enabled
        has_offscreen_renderer=False,  # Disable offscreen rendering for visible output
        use_camera_obs=False,
    )
    print("Environment successfully initialized!")
except Exception as e:
    print(f"Failed to initialize environment: {e}")
    exit(1)

# Step 2: Wrap the environment for Gym compatibility
try:
    print("Wrapping the environment with GymWrapper...")
    my_wrapped_env = GymWrapper(env)
    my_vec_env = DummyVecEnv([lambda: my_wrapped_env])
    my_vec_env = VecNormalize(my_vec_env, norm_obs=True, norm_reward=True)
    print("Environment wrapped successfully!")
except Exception as e:
    print(f"Failed to wrap environment: {e}")
    exit(1)

# Step 3: Define a trajectory (e.g., a list of desired joint angles or end-effector positions)
desired_trajectory = [
    [0.1, 0.2, 0.3],  # Step 1 positions
    [0.15, 0.25, 0.35],  # Step 2 positions
    [0.2, 0.3, 0.4],  # Step 3 positions
    # I need to create movement trajectory within my run_Script or in teh lift env? 
]

# Step 4: Modify the reward function to encourage following the trajectory
def custom_reward_function(obs, action, step_count):
    current_position = obs['robot0_proprio-state'][:3] # enhance this later ??
    target_position = desired_trajectory[min(step_count, len(desired_trajectory) - 1)]
    distance_to_trajectory = np.linalg.norm(np.array(current_position) - np.array(target_position))
    reward = -distance_to_trajectory * 10  # Penalize deviations from the trajectory
    forward_movement = current_position[0]
    reward += forward_movement * 10  # Encourage forward movement
    return reward

# Step 5: Initialize PPO model
try:
    print("Initializing the PPO model...")
    modelInitialize = PPO(
        policy="MlpPolicy",
        env=my_vec_env,
        learning_rate=0.0003,
        n_steps=3000,
        batch_size=500,
        verbose=1,
        tensorboard_log='/home/anastasiia/tb.log/'
    )
    print("PPO model initialized successfully!")
except Exception as e:
    print(f"Failed to initialize PPO model: {e}")
    exit(1)

# Step 6: Explicitly train the model
try:
    print("Starting the training process...") # print to check if it works
    modelInitialize.learn(
        total_timesteps=50000,
        log_interval=1,
    )
    print("Training completed successfully!")
    modelInitialize.save("trajectory_trained_model")
except Exception as e:
    print(f"Failed during training: {e}")
    exit(1)

# Step 7: Run the trained model to observe the learned behavior
obs = env.reset()
env.render()  # Ensure rendering starts after reset
step_count = 0
for i in range(1000):
    obs_array = obs['robot0_proprio-state']

    # Check if padding is needed to match the expected observation space (e.g., 42 elements)
    if obs_array.shape[0] < 42:
        obs_array = np.pad(obs_array, (0, 42 - obs_array.shape[0]), 'constant')

    action, _states = modelInitialize.predict(obs_array)
    obs, reward, done, info = env.step(action)

    # Use the custom reward function based on the trajectory
    modified_reward = custom_reward_function(obs, action, step_count)
    reward += modified_reward

# # including for no reason? 
# for i in range (5000):
#     obs_array = obs['robot0_proprio-state']

    env.render()  # Ensure the simulation renders at each step

    step_count += 1
    if done:
        obs = env.reset()
        env.render()  # Render again after resetting
        step_count = 0