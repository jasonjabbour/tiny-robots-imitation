
"""This file implements the Gym Environment"""

import collections
import time
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pybullet
import pybullet_utils.bullet_client as bullet_client
import pybullet_data as pd

from envs.sensors import sensor
# Using instead of robot config:
from robots import motor_control_config

_NUM_SIMULATION_ITERATION_STEPS = 300

class TinyRobotGymEnv(gym.Env):
    """Open AI Gym Environment for Locomotion Tasks"""

    #Store additional information about gym environment class
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 100
    }

    def __init__(self,
                 gym_config,
                 robot_class = None,
                 sensors=None, 
                 task=None,
                 env_randomizers=None):
        """Initialize Open AI Gym Environment
        
        Args:
            gym_config: An instance of TinyRobotGymConfig
            robot: A class of a robot. We provide a class rather than an 
                instance due to hard_reset functionality. Parameters are expected to be
                configured with gin
            sensors: A list of environmental and robot sensors for observations
            task: A callable function/class to calculate the reward and termination
                condition. Takes the gym env as the argument when calling. 
            env_randomizers: A list of EnvRandomizers(s). An EnvRandomizer may
                randomize the physical property of the robot, change the terrain during
                reset(), or add perturbation forces during step()
        
        Raises:
            ValueError: If the num_action_repeat is less than 1       
        
        """

        self.seed()
        self._gym_config = gym_config
        self._robot_class = robot_class
        self._sensors = sensors if sensors is not None else list()
        self._env_randomizers = env_randomizers if env_randomizers is not None else list()

        # A dictionary containing the objects in the world other than the robot
        self._world_dict = {}
        self._task = task

        # Sanity Check robot class is passed in
        if self._robot_class is None:
            raise ValueError('Robot class cannot be None.')

        # This is a workaround due to the issue in b/130128505#comment5
        if isinstance(self._task, sensor.Sensor):
            self._sensors.append(self._task)

        # Simulation related parameters
        self._num_action_repeat = gym_config.simulation_parameters.num_action_repeat
        self._on_rack = gym_config.simulation_parameters.robot_on_rack
        if self._num_action_repeat < 1:
            raise ValueError('number of action repeats should be at least 1')
        # Set time steps for sim and env
        self._sim_time_step = gym_config.simulation_parameters.sim_time_step_s
        # FIX
        self._env_time_step = self._num_action_repeat * self._sim_time_step #33 * .001s
        self._env_step_counter = 0

        # 300 / 1 = 300 
        self._num_bullet_solver_iterations = int(_NUM_SIMULATION_ITERATION_STEPS/self._num_action_repeat)
        self._is_render = gym_config.simulation_parameters.enable_rendering

        # The wall-clock time at which the last frame is rendered
        self._last_frame_time = 0.0
        # Debug Visualizer Slider ID 
        self._show_reference_id = -1

        # Initialize Pybullet using GUI or Direct
        if self._is_render:
            # Pybullet GUI
            self._pybullet_client = bullet_client.BulletClient(
                connection_mode=pybullet.GUI)
            # Debug Visualizer on GUI
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_GUI, 
                gym_config.simulation_parameters.enable_rendering_gui)
            # Show goal robot if in reward task
            if hasattr(self._task, '_draw_ref_model_alpha'):
                # Add Slider for showing goal robot with an transperance value from 0 to 1. Default alpha in _task
                self._show_reference_id = pybullet.addUserDebugParameter('show reference',0,1,
                    self._task._draw_ref_model_alpha)
            # Add Slider for slowing visualization down 0 sec - .3 sec. Default 0 seconds delay
            self._delay_id = pybullet.addUserDebugParameter('delay', 0, 0.3, 0)
        # Do not render GUI
        else:
            # Pybullet DIRECT
            self._pybullet_client = bullet_client.BulletClient(
                connection_mode=pybullet.DIRECT)
        
        # Set additional pybullet directory
        self._pybullet_client.setAdditionalSearchPath(pd.getDataPath())

        # For hardware OpenGL acceleration rendering (or used in Google Colab plugins)
        if gym_config.simulation_parameters.egl_rendering:
            self._pybullet_client.loadPlugin('eglRendererPlugin')
        
        # Build the Gym action space depending on motor mode type
        self._build_action_space()

        # Set camera parameters
        self._camera_dist = gym_config.simulation_parameters.camera_distance
        self._camera_yaw = gym_config.simulation_parameters.camera_yaw
        self._camera_pitch = gym_config.simulation_parameters.camera_pitch
        self._render_width = gym_config.simulation_parameters.render_width
        self._render_height = gym_config.simulation_parameters.render_height
        self._hard_reset = True

        # Call reset to prepare environment
        self.reset()

        # Now follow hard reset requirement
        self._hard_reset = gym_config.simulation_parameters.enable_hard_reset

    
    def _build_action_space(self):
        """Build action space based on motor control mode"""

        # Get motor type (Position, Torque, Hybrid, ...)
        motor_mode = self._gym_config.simulation_parameters.motor_control_mode

        # Actions are motor positions (also valid for pwm mode)
        if motor_mode == motor_control_config.MotorControlMode.POSITION: 

            action_upper_bound = []
            action_lower_bound = []

            # Upper and Lower bounds set in robot class for each joint
            action_config = self._robot_class.ACTION_CONFIG

            # Each action is a tiny_robot_gym_config ScalarField
            for action in action_config:
                # Add the upper and lower bound of each joint to list
                action_upper_bound.append(action.upper_bound)
                action_lower_bound.append(action.lower_bound)
            
            # Create the gym action space
            self.action_space = spaces.Box(np.array(action_lower_bound), 
                                                    np.array(action_upper_bound), 
                                                    dtype=np.float32)
        else:
            raise ValueError('Motor mode not supported. Some motor modes have been \
            omitted for simplicity. Please see the original implementation: \
            https://github.com/jasonjabbour/motion_imitation/blob/master/motion_imitation/envs/locomotion_gym_env.py')

    def close(self):
        if hasattr(self, '_robot') and self._robot:
            self._robot.Terminate()
    
    def seed(self, seed=None):
        self.np_random, self.np_random_seed = seeding.np_random(seed)
        return [self.np_random_seed]

    def all_sensors(self):
        """Returns all robot and environmental sensors."""
        return self._robot.GetAllSensors() + self._sensors

    def sensor_by_name(self, name):
        """ Returns the sensor with the given name, or None if does not exist."""
        for sensor_ in self.all_sensors():
            if sensor_.get_name() == name:
                return sensor_

    def reset(self,
              initial_motor_angles=None,
              reset_duration=0.0,
              reset_visualization_camera=True):
        """Resets the robot's position in the world or rebuild the sim world.

        The simulation world will be rebuilt if self._hard_reset is True.

        Args:
            initial_motor_angles: A list of Floats. The desired joint angles after
                reset. If None, the robot will use its built-in value.
            reset_duration: Float. The time (in seconds) needed to rotate all motors
                to the desired initial values.
            reset_visualization_camera: Whether to reset debug visualization camera on
                reset.

        Returns:
            A numpy array contains the initial observation after reset.        
        """
        # Disable Rendering during reset
        if self._is_render:
            self._pybullet_client.configureDebugVisualizer(
                self._pybullet_client.COV_ENABLE_RENDERING, 0)
        
        # Clear the simulation world and rebuild the robot interface
        if self._hard_reset:
            # Remove all objects from world and reset the world conditions
            self._pybullet_client.resetSimulation()
            # Set maximum number of constraint solver iterations
            self._pybullet_client.setPhysicsEngineParameter(
                numSolverIterations=self._num_bullet_solver_iterations)
            # Default 240 Hz between each step. 
            # In many cases it is best to leave the default
            # Several parameters must be duned including number of solver iterations and 
            # error reduction parameters (erp) for contact friction and non-contact joints
            # It is best to change the timestep when calling stepSimulation
            self._pybullet_client.setTimeStep(self._sim_time_step)

            # FIX ^^try removing setTimeStep and numSolverIterations and see what happens with default

            # Set gravity
            self._pybullet_client.setGravity(0,0,-10)

            # Rebuild the World and add ground to dict
            self._world_dict = {
                'ground': self._pybullet_client.loadURDF('plane_implicit.urdf')
            }

            # Rebuild the robot and create the _robot object
            self._robot = self._robot_class(
                pybullet_client=self._pybullet_client,
                sensors=self._sensors, 
                on_rack=self._on_rack, 
                action_repeat=self._gym_config.simulation_parameters.num_action_repeat,
                motor_control_mode=self._gym_config.simulation_parameters.motor_control_mode, 
                reset_time=self._gym_config.simulation_parameters.reset_time, 
                enable_clip_motor_commands=self._gym_config.simulation_parameters.enable_clip_motor_commands, 
                enable_action_filter=self._gym_config.simulation_parameters.enable_action_filter, 
                enable_action_interpolation=self._gym_config.simulation_parameters.enable_action_interpolation, 
                allow_knee_contact=self._gym_config.simulation_parameters.allow_knee_contact, 
                dead_zone=self._gym_config.simulation_parameters.dead_zone)

        # Reset the pose of the robot.
        self._robot.Reset(reload_urdf=False, 
                          default_motor_angles=initial_motor_angles, 
                          reset_time=reset_duration)

        # Disable Cone Friction in Pybullet
        self._pybullet_client.setPhysicsEngineParameter(enableConeFriction=0)

        # Reset Counter
        self._env_step_counter = 0

        # Reset Camera Position
        if reset_visualization_camera:
            self._pybullet_client.resetDebugVisualizerCamera(self._camera_dist, 
                                                             self._camera_yaw, 
                                                             self._camera_pitch,
                                                             [0, 0, 0])
        
        # Reset saved last action. Shape is (8,)
        self._last_action = np.zeros(self.action_space.shape)

        # Re-enable rendering since reset is done
        if self._is_render:
            self._pybullet_client.configureDebugVisualizer(
                self._pybullet_client.COV_ENABLE_RENDERING, 1)
        
        # Reset all sensors.
        # HistoricSensorWrapper creates a new buffer of zeroz and gets one new observation
        for s in self.all_sensors():
            s.on_reset(self)
        
        # Reset Task
        if self._task and hasattr(self._task, 'reset'):
            self._task.reset(self)

        # Loop over all env randomizers and randomize environment
        for env_randomizer in self._env_randomizers:
            env_randomizer.randomize_env(self) 

        # Return the observation at this step
        return self._get_observation()

    def step(self, action):
        """Step forward the simulation, given the action.
        
        Args:
            action: Can be a list of desired motor angles for all motors when the
                robot is in position control mode. The action must be compatible with
                the robot's motor control mode. We are not going to use the leg space
                (swing/extension) definition at the gym level, since they are specific to
                the TinyRobot.

        Returns:
            observations: The observation dictionary. The keys are the sensor names
                and the values are the sensor readings.
            reward: The reward for the current state-action pair.
            done: Whether the episode has ended.
            info: A dictionary that stores diagnostic information
        
        Raises:
            ValueError: The action dimension is not the same as the number of motors.
            ValueError: The magnitude of actions is out of bounds        
        """
        # Get current position of robot
        self._last_base_position = self._robot.GetBasePosition()
        # Save action predicted by NN
        self._last_action = action

        if self._is_render:
            # Sleep, otherwise the computation takes less time than real time, 
            # which will make the visualization like a fast-forward video.
            time_spent = time.time() - self._last_frame_time
            self._last_frame_time = time.time()
            time_to_sleep = self._env_time_step - time_spent

            # Make sure to sleep until computation is over then render
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)

            # FIX what happens if you use stepsimulation instead ^^
            
            # Get position of robot
            base_pos = self._robot.GetBasePosition()

            # Keep the previous orentation of the camera set by the user
            [yaw, pitch, dist] = self._pybullet_client.getDebugVisualizerCamera()[8:11]
            # Advance camera to current position of robot
            self._pybullet_client.resetDebugVisualizerCamera(dist, yaw, pitch, base_pos)

            # Smooth Simulation Rendering 
            self._pybullet_client.configureDebugVisualizer(
                self._pybullet_client.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
            
            # Poll for alpha slider value
            alpha = 1.
            if self._show_reference_id>=0:
                alpha = self._pybullet_client.readUserDebugParameter(self._show_reference_id)

            # Draw reference robot
            ref_col = [1, 1, 1, alpha] #White color
            if hasattr(self._task, '_ref_model'):
                # Make the robot a different color
                self._pybullet_client.changeVisualShape(self._task._ref_model, -1, rgbaColor=ref_col)
                # Iterate through each joint and change color
                for l in range (self._pybullet_client.getNumJoints(self._task._ref_model)):
                    self._pybullet_client.changeVisualShape(self._task._ref_model, l, rgbaColor=ref_col)
            
            # Poll for delay slider value
            delay = self._pybullet_client.readUserDebugParameter(self._delay_id)

            # Slow the visualization down if needed by sleeping
            if delay>0:
                time.sleep(delay)

        # Randomize environment for the next step
        for env_randomizer in self._env_randomizers:
            env_randomizer.randomize_step(self)

        # Processed action by robot
        proc_action = self._robot.Step(action)

        # HistoricSensorWrapper appends new observation to sensor history buffer
        for s in self.all_sensors():
            s.on_step(self)

        # Update task
        if self._task and hasattr(self._task, 'update'):
            self._task.update(self)   

        # Call reward function
        reward = self._reward()

        # Check if episode is done
        done = self._termination()

        # Increment step
        self._env_step_counter +=1

        # End episode if done
        if done:
            self._robot.Terminate()
        
        return self._get_observation(), reward, done, {"processed_action":proc_action}

    def render(self, mode='rgb_array'):
        """Camera Rendering
        
        Calculate the view matrix and the projection matrix

        More information: 
            http://ksimek.github.io/2013/08/13/intrinsic/
        
        """
        # Validate default mode
        if mode != 'rgb_array':
            raise ValueError('Unsupported render mode:{}'.format(mode))
        
        #Get position of robot
        base_pos = self._robot.GetBasePosition()

        # View Matrix
        view_matrix = self._pybullet_client.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=self._camera_dist,
            yaw=self._camera_yaw,
            pitch=self._camera_pitch,
            roll=0,
            upAxisIndex=2)
        
        # Projection Matrix
        proj_matrix = self._pybullet_client.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(self._render_width) / self._render_height,
            nearVal=0.1,
            farVal=100.0)
        
        # Get Camera Image (list of pixel colors in RGBA format [0..255] for each color) 
        (_, _, px, _, _) = self._pybullet_client.getCameraImage(
            width=self._render_width,
            height=self._render_height,
            renderer=self._pybullet_client.ER_BULLET_HARDWARE_OPENGL,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix)
        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]

        # Return list of pixels
        return rgb_array

    def get_ground(self):
        """Get simulation ground model."""
        return self._world_dict['ground']
    
    def set_ground(self, ground_id):
        """Set simulation ground model."""
        self._world_dict['ground'] = ground_id

    @property
    def rendering_enabled(self):
        return self._is_render

    @property
    def last_base_position(self):
        return self._last_base_position

    @property
    def world_dict(self):
        return self._world_dict.copy()

    @world_dict.setter
    def world_dict(self, new_dict):
        self._world_dict = new_dict.copy()
    
    def _termination(self):
        if not self._robot.is_safe:
            return True
        
        if self._task and hasattr(self._task, 'done'):
            return self._task.done(self)
        
        # Tracing from sensor_wrappers.py -> robot_sensors.py -> sensors.py, on_terminate does nothing
        for s in self.all_sensors():
            s.on_terminate(self)

        return False
    
    def _reward(self):
        """Reward is returned from the task"""
        if self._task:
            return self._task(self)
        return 0

    def _get_observation(self):
        """Get observation of this environment
        
        Returns:
            observations: sensory observation in the numpy array format        
        """
        sensors_dict = {}

        # all_sensors is a list of a BoxSensor wrapped by a HistoricSensorWrapper
        for s in self.all_sensors():
            # get observation returns the already created history buffer
            # a new observation is observed during the on_step() function
            # see sensor_wrappers.py HistoricSensorWrapper
            sensors_dict[s.get_name()] = s.get_observation()

        # Create a sorted collection of sensor history buffers
        observations = collections.OrderedDict(sorted(list(sensors_dict.items())))

        return observations
        
    @property
    def pybullet_client(self):
        return self._pybullet_client

    @property
    def robot(self):
        return self._robot

    @property
    def env_step_counter(self):
        return self._env_step_counter

    @property
    def hard_reset(self):
        return self._hard_reset

    @property
    def last_action(self):
        return self._last_action

    @property
    def env_time_step(self):
        return self._env_time_step

    @property
    def task(self):
        return self._task

    @property
    def robot_class(self):
        return self._robot_class