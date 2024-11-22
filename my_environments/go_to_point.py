from collections import OrderedDict

print("benu GO TO IT")

import numpy as np

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
#from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat

from collections import OrderedDict

from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat


'''
Adding stuff from my_model folder
'''
from my_model.tasks import GoToPointTask
from my_model.arenas import EmptyArena


class GoToPointTask(SingleArmEnv):
    """
    This class corresponds to the lifting task for a single robot arm.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        placement_initializer (ObjectPositionSampler): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

        camera_segmentations (None or str or list of str or list of list of str): Camera segmentation(s) to use
            for each camera. Valid options are:

                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level

            If not None, multiple types of segmentations can be specified. A [list of str / str or None] specifies
            [multiple / a single] segmentation(s) to use for all cameras. A list of list of str specifies per-camera
            segmentation setting(s) to use.

    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        # table_full_size=(0.8, 0.8, 0.05),
        # table_friction=(1.0, 5e-3, 1e-4),
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
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mujoco",
        renderer_config=None,
    ):
        # settings for table top
        # self.table_full_size = table_full_size
        # self.table_friction = table_friction
        # self.table_offset = np.array((0, 0, 0.8))

        # I added controlers because nothing worked 
        if controller_configs is None:
            controller_configs = {
                "type": "OSC_POSITION",  # cartesian position controller
                "input_max": 1,
                "input_min": -1,
                "output_max": 0.5,
                "output_min": -0.5,
                "kp": 150,  # can reduce to 50 to solw down I guess
                "damping_ratio": 1,  # for smooth motion
                "impedance_mode": "fixed",  # fixed stiffness
                "kp_limits": [0, 300], 
                "damping_ratio_limits": [0, 10],
                "position_limits": None,  # no cartesian position limits
                "orientation_limits": None,  # no cartesian orientation limits
                "interpolation": "linear",  # this line here I need I guess
                "ramp_ratio": 0.2,
            }


        if placement_initializer is None:
            placement_initializer = UniformRandomSampler(
            name="ObjectSampler",
            mujoco_objects=None,  # Start with no objects
            x_range=[-0.2, 0.2],
            y_range=[-0.2, 0.2],
            rotation=None,
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            z_offset=0.01,
         )
            
        self.placement_initializer = placement_initializer

        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs


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
                    # reward configuration

    # # I commented out original reward function to avoid overriding and wrong results
    # def reward(self, action=None):
    #     """
    #     Reward function for the task.
    #     Sparse un-normalized reward:
    #         - a discrete reward of 2.25 is provided if the cube is lifted
    #     Un-normalized summed components if using reward shaping:
    #         - Reaching: in [0, 1], to encourage the arm to reach the cube
    #         - Grasping: in {0, 0.25}, non-zero if arm is grasping the cube
    #         - Lifting: in {0, 1}, non-zero if arm has lifted the cube
    #     The sparse reward only consists of the lifting component.
    #     Note that the final reward is normalized and scaled by
    #     reward_scale / 2.25 as well so that the max score is equal to reward_scale
    #     Args:
    #         action (np array): [NOT USED]
    #     Returns:
    #         float: reward value
    #     """
    #     reward = 0.0
    #     # sparse completion reward
    #     if self._check_success():
    #         reward = 2.25
            
        
    # added reward based on how close the gripper is to the target
    # task is to ensure the loss is lowered ==> will adjust reward accordingly

    def reward(self, action):
        gripper_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
        distance = np.linalg.norm(gripper_pos - self.current_target_position)
        reward = 1 - np.tanh(5.0 * distance) # from 10.0 to 5.0 how close gripper to tgt

        # I added new elemenents to the reward to ensure precision ==> need to know if it needs to be
        previous_distance = getattr(self, "previous_distance", float("inf")) #getattr used in case prev_distance attribute doesn't exist
        if distance < previous_distance:
                reward += 0.1  # more points for getting closer to teh target
        self.previous_distance = distance

        if distance < 0.05:
            reward += 10.0  # higher encouragment

        # remember in case need exponent the reward
        # reward -= 0.01 * self.current_step  # <== small penalty that will increase as we progresss
        # self.current_step += 1

        return reward
    

    def set_target_position(self, target_position):
        # using this to update the current target position
        self.current_target_position = target_position
        

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = np.array([0, 0, 0.4])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = EmptyArena()

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # initialize objects of interest
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        self.cube = BoxObject(
            name="cube",
            size_min=[0.020, 0.020, 0.020],  # [0.015, 0.015, 0.015],
            size_max=[0.022, 0.022, 1.022],  # [0.018, 0.018, 0.018])
            rgba=[1, 0, 0, 1],
            material=redwood,
        )

        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.cube)
        else:
            self.placement_initializer = UniformRandomSampler(
            name="ObjectSampler",
            mujoco_objects=self.cube,
            x_range=[-0.2, 0.2],
            y_range=[-0.2, 0.2],
            rotation=None,
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=[0.0, 0.5, 0.02],
            z_offset=0.01,
        )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.cube,
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.cube_body_id = self.sim.model.body_name2id(self.cube.root_body)

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        ###########################################################################################
        # adding gripper information for the position sensor
        @sensor(modality="robot")
        def gripper_pos(obs_cache):  # from facts I can update?
            return np.array(self.sim.data.site_xpos[self.robots[0].eef_site_id]) # taken form here
        
         # adding gripper position observable 
        observables["gripper_pos"] = Observable(
        name = "gripper_pos",
        sensor = gripper_pos,
        sampling_rate = self.control_freq, # on its own
        )

        # here I can define gripper to cube distance to ease the tracking process on how far is ot form the cube
        def gripper_to_cube_pos(obs_cache):
            if "gripper_pos" in obs_cache and "cube_pos" in obs_cache:  # get hand and get cube pos
                return np.linalg.norm(obs_cache["gripper_pos"] - obs_cache["cube_pos"])
        
            return np.zeros(1)  # if robot doesn't know ehre the cube is return 0 array size 1

        # low-level object information
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"

            # cube-related observables
            @sensor(modality=modality)
            def cube_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.cube_body_id])

            @sensor(modality=modality)
            def cube_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.cube_body_id]), to="xyzw")

            @sensor(modality=modality)
            def gripper_to_cube_pos(obs_cache):
                return (
                    obs_cache[f"{pf}eef_pos"] - obs_cache["cube_pos"]
                    if f"{pf}eef_pos" in obs_cache and "cube_pos" in obs_cache
                    else np.zeros(3)
                )

            sensors = [cube_pos, cube_quat, gripper_to_cube_pos]
            names = [s.__name__ for s in sensors]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        # here I will return updated information on the observables dictionary 
        return observables

        ############################################################################################
    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        self.current_step = 0

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

        # here we ensure that target position is set each time when env resets internally #
        target_position = np.array([0.5, 0.0, 1.0])  # can change value if needed, let me know if need random
        self.set_target_position(target_position)
        #### I made changes here! ####
        # finding gripper's starting pos
        start_position = np.array(self.sim.data.site_xpos[self.robots[0].eef_site_id])
        # attmpting to create a trajectory/path from start to target
        self.generated_trajectory = self.generate_trajectory(
        start=start_position,
        target=target_position,
        num_steps=50  # number of positions in the path
)

    def generate_trajectory(self, start, target, num_steps):  # num_steps = 100 to ensure less fast I guess
        trajectory = [
            [0.5, 0.0, 1.0],  # first point
            # [0.4, -0.1, 1.2],  # second point  negative values to move left
            # [0.3, 0.1, 1.1],   # third point
            # [0.2, -0.2, 1.3],  # fourth point   idk if its what needed 
    ]

        return np.array(trajectory)

    # or can do these too
    #     hardcoded_points = [
    #     [0.5, 0.0, 1.0],  # First point
    #     [0.4, -0.1, 1.2],  # Second point
    # ]

    # # generate points between the last hardcoded point and the target
    # if start is not None and target is not None:
    #     dynamic_points = np.linspace(hardcoded_points[-1], target, num_steps)
    #     return np.concatenate([hardcoded_points, dynamic_points], axis=0)
    
    # # If no dynamic points are needed, just return hardcoded ones
    # return np.array(hardcoded_points)
        # slowly getting from start to teh target

        # return np.linspace(start, target, num_steps)
    
    #this one is my control element ==> must follow points generated in generate trajectiry
    def move_along_trajectiry_waypoints(self):
        """
        Moves the robot along the generated trajectory step by step.
        """
        # If there are still waypoints to follow
        if self.current_step < len(self.generated_trajectory):
            # get the next waypoint position
            next_position_waypoint = self.generated_trajectory[self.current_step]
            # print the target and current positions
            print(f"Step {self.current_step}: Moving to {next_position_waypoint}")
            print(f"Current Position: {self.sim.data.site_xpos[self.robots[0].eef_site_id]}")
            action = np.concatenate((next_position_waypoint, [0.0]))  # [x, y, z, gripper_action]
            self.robots[0].control(action)
            # controller to move toward the next waypoint
            self.robots[0].control(action)
            # increment to the next waypoint
            self.current_step += 1
        # get the next waypoint position
            next_position_waypoint = self.generated_trajectory[self.current_step]

        # print the target and current positions
            print(f"Step {self.current_step}: Moving to {next_position_waypoint}")
            print(f"Current Position: {self.sim.data.site_xpos[self.robots[0].eef_site_id]}")
        #  controller to move toward the next waypoint
            #self.robots[0].controller.set_action({"position": next_position_waypoint})
        # increment to the next waypoint
            self.current_step += 1

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the cube.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # added thsi function here to move along the path###
        self.move_along_trajectiry_waypoints() 

        # run superclass method first
        super().visualize(vis_settings=vis_settings)

        # color the gripper visualization site according to its distance to the cube | is it needed??
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.cube)

    def _check_success(self):
        """
        Check if cube has been lifted.

        Returns:
            bool: True if cube has been lifted
        """
        return False
        cube_height = self.sim.data.body_xpos[self.cube_body_id][2]
        table_height = self.model.mujoco_arena.table_offset[2]

        # cube is higher than the table top above a margin
        # print("check_success called")
        if( cube_height > table_height + 0.04):
            print("Cube lifted")
        return cube_height > table_height + 0.04
