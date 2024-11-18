import robosuite as suite
from robosuite.wrappers import GymWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from robosuite.utils.placement_samplers import UniformRandomSampler
from my_environments import GoToPointTask

def make_env():
    placement_initializer = UniformRandomSampler(
        name="object_placement_sampler",
        x_range=(0.1, 0.2),
        y_range=(0.1, 0.2),
        ensure_object_boundary_in_range=False,
        ensure_valid_placement=True,
        reference_pos=(0.0, 0.0, 0.8),
        z_offset=0.01,
    )
    env = suite.make(
        env_name="GoToPointTask",
        robots="Panda",
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        placement_initializer=placement_initializer,
    )
    env_gym = GymWrapper(env)
    env = DummyVecEnv([lambda: env_gym])

##############################################################################################################
    # This can help if problems with loading 
    vec_normalize_path = Path("./training_models/vec_normalize.pkl") # path to Vec Normalize as on lab machine
    
    if vec_normalize_path.is_file() and vec_normalize_path.stat().st_size > 0: # checking if file exists simple stuff
        try:
            env = VecNormalize.load(vec_normalize_path, env) # load this vecNormalize
            print("VecNormalize loaded successfully.")   # printing just to know if it worked
        except EOFError: # if file not working 
            print("VecNormalize doesn't exist. Create new VecNormalize.")
            env = VecNormalize(env) # just not working
    else:
        print("VecNormalize not found. Creating new VecNormalize.")
        env = VecNormalize(env)

    # Check if the model file exists and is valid
    model_path = Path("./training_models/point_model.zip") # just put path and it will be set

    if model_path.is_file() and model_path.stat().st_size > 0: # checking if it exists
        try:
            model = PPO.load(model_path, env=env)
            print("Model loaded successfully.") # success
        except ValueError:
            print("Model doesn't exist. Creating a new model.")
            model = PPO("MlpPolicy", env, verbose=1) # not success
    else:
        print("Model not found. Creating a new model.")
        model = PPO("MlpPolicy", env, verbose=1) # need to create model if no model found

##############################################################################################################
    
    # Reset the environment and begin interaction
    obs = env.reset()
    print("Initial observation:", obs)

    # run model this number of steps if needed
    for i in range(100000):
        action, _states = model.predict(obs)

        obs, reward, done, info = env.step(action)
        

        # I think this helps to keep track of data
        print(f"Reward: {reward}, Done: {done}, Info: {info}") # Adjust this based on what step() returns

        env_gym.render()

        if done:
            obs = env.reset() # Reset environment if done

