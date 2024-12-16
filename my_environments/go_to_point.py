from collections import OrderedDict
import numpy as np
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from my_model.tasks import GoToPointTask
from my_model.arenas import EmptyArena


class GoToPointTask(SingleArmEnv):
    def step(self, action): #! need to override this function to ensure the robot moves towards the target
        # Clip the action for safety
        clipped_action = np.clip(action, -0.5, 0.5)
        return super().step(clipped_action)
    
    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=4000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mujoco",
        renderer_config=None,
        listofCord = [np.array([0.5, 0.5, 0.5]), np.array([.71, 0.71, .71]), np.array( [0.3,  0.6,  1.3]), np.array([0.55, 0.55, 0.55])],
        index = 0,
        target_coordinate= np.array([0.5, 0.5, 0.5]),  # default target position
        rewardindex = 0,
        
    ):
        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.current_step = 0
        self.rewardindex = rewardindex

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs
        self.baseline_reward = 0.1

        # object placement initializer
        self.placement_initializer = placement_initializer
        self.target_coordinate = target_coordinate
        self.listofCord = listofCord
        self.index = index

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )
    def reward(self, action):
        
        gripper_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
        distance = np.linalg.norm(gripper_pos - self.target_coordinate)
        # self.current_step += 1
        reward = float(1 - np.tanh(distance))
        # I added new elemenents to the reward to ensure precision ==> need to know if it needs to be
        previous_distance = getattr(self, "previous_distance", float("inf")) #getattr used in case prev_distance attribute doesn't exist
        if distance < previous_distance:
                reward *= 1.1   # more points for getting closer to teh target
        previous_distance = distance
        # less points for moving away from the target
        self.rewardindex += 1
        reward = (reward  + self.baseline_reward ) -reward * (self.rewardindex/ (200 + self.rewardindex))  # the longer it takes to reach the target, the less the reward 
        
        if (self.rewardindex % 500 == 0):
            print(f"current distance:{distance:.5f} the current cord {gripper_pos}, stuck at cord {self.target_coordinate}")
        if distance < 0.2:
            self.next_cord()
            if (self.index == 3):
                self.done = True
                print("All targets reached")
                return reward * 10
            # print("Target reached: New cordinate", self.target_coordinate
            self.baseline_reward += reward
            
            print("New target coordinate: ", self.target_coordinate)
            reward *= 10 # more points for reaching the target
        return reward
    
    def next_cord(self):
        self.rewardindex = 0
        self.index += 1
        self.target_coordinate =  self.listofCord[self.index % 4]
        self.previous_distance = distance = np.linalg.norm(self.sim.data.site_xpos[self.robots[0].eef_site_id] - self.target_coordinate)

    def set_target_position(self, target_position):
        # using this to update the current target position
        self.target_coordinate = target_position
        

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = np.array([0, 0, 0])
        self.robots[0].robot_model.set_base_xpos(xpos)
        #self.robots[0].gripper.set_base_xpos(xpos)
        
        # load model for table top workspace
        mujoco_arena = EmptyArena()

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        #self.cube_body_id = self.sim.model.body_name2id(self.cube.root_body)

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"
            @sensor(modality=modality)
            def time(obs_cache):
                return self.index
            
            @sensor(modality=modality)
            def currentPOS(obs_cache):
                return self.sim.data.site_xpos[self.robots[0].eef_site_id]
            @sensor(modality=modality)
            def currentTarget(obs_cache):
                return self.target_coordinate
            
            @sensor(modality=modality)
            def distance(obs_cache):
                return (
                     np.linalg.norm(self.sim.data.site_xpos[self.robots[0].eef_site_id] - self.target_coordinate)
                )


            sensors = [currentPOS, time, currentTarget, distance]
            names = [s.__name__ for s in sensors]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        self.current_step = 0
        self.index = 0
        self.rewardindex = 0
        self.target_coordinate = self.listofCord[self.index % 4]
        self.baseline_reward = 0.1


    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the cube.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the cube
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.cube)

