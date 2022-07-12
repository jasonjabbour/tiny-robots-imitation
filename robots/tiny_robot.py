"""This file implements the functionalities of a Tiny Toboy using pybullet"""

import os
import inspect
from turtle import onrelease
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import collections
import copy
import math
import re
import numpy as np

from robots import math_conversion_utils
from robots import motor_control_config

SENSOR_NOISE_STDDEV = (0.0, 0.0, 0.0, 0.1, 0.0) #NEW was all zeroz
DEFAULT_MOTOR_DIRECTIONS = (1, 1, 1, 1, 1, 1, 1, 1) # use -1 to denote oposite direction
DEFAULT_MOTOR_OFFSETS = (0, 0, 0, 0, 0, 0, 0, 0)


class TinyRobot(object):
    """The TinyRobot Class that simulates a tiny quadruped robot"""
    def __init__(self,
                 pybullet_client, 
                 num_motors, 
                 dofs_per_leg, 
                 time_step=0.01, 
                 action_repeat=1, 
                 self_collision_enabled=False, 
                 motor_control_mode=motor_control_config.MotorControlMode.POSITION, 
                 motor_model_class=None, 
                 motor_kp=None, 
                 motor_kd=None, 
                 motor_torque_limits=None, 
                 pd_latency=0.0, 
                 control_latency=0.0, 
                 observation_noise_stdev=SENSOR_NOISE_STDDEV,
                 motor_overheat_protection=False, 
                 motor_direction=DEFAULT_MOTOR_DIRECTIONS, 
                 motor_offset=DEFAULT_MOTOR_OFFSETS, 
                 on_rack=False, 
                 reset_at_current_position=False, 
                 sensors=None, 
                 enable_action_interpolation=False, 
                 enable_action_filter=False, 
                 reset_time=-1, 
                 dead_zone=False):
        """Tiny Robot Constructor
        
        Args:
            pybullet_client: The instance of BulletClient to manage different simulations.
            num_motors: The number of motors on the robot.
            dofs_per_leg: The number of degrees of freedom for each leg. 
            time_step: the time step of the simulation
            action_repeat: The number of ApplyAction() for each control step.
            self_collision_enabled: Whether to enable self collision
            motor_control_mode: Enum. Can either by POSITION, TORQUE, or HYBRID. 
            motor_model_class: We can choose from simple pd model to more accruate DC
                motor models. 
            motor_kp: proportional gain for the motors
            motor_kd: derivative gain for the motors
            motor_torque_limits: Torque limits for the motors. Can be a single float
                or a list of floats specifying different limits for different robots. If 
                not provided, the default limit of the robot is used. 
            pd_latency: The latency of the observations (in seconds) used to calculate action. 
                On the real hardware, it is the latency from the motor controller, 
                the microcontroller to the host (Nvidia TX2, Raspberry Pi, ...)
            observation_noise_stdev: The standard deviation of a Gaussian noise model for 
                the sensor. It should be an array for separate snesors in the
                following order [motor_angle, motor_velocity, motor_torque, 
                base_roll_pitch_yaw, base_angular_velocity]
            motor_overheat_protection: Whether to shutdown the motor that has exerted
                large torque (OVERHEAT_SHUTDOWN_TORQUE) for an extended amount of time
                (OVERHEAT_SHUTDOWN_TIME). See ApplyAction() in tiny_robot.py for more details
            motor_direction: A list of direction values, either 1 or -1, to compensate
                the axis difference of motors between the simulation and the real robot
            motor_offset: A list of offset value for the motor angles. This is used
                to compensate the angle difference between the simulation of the real robot. 
            on_rack: Whether to place the robot on rack. This is only used to debug the walking
                gait. In this mode, the robot's base is hanged midair so that its walking 
                gait is clearer to visualize
            reset_at_current_position: Whether to reset the robot at the current
                position and orientation. This is for simulating the reset behavior 
                in the real world. 
            sensors: a list of sensors that are attached to the robot. 
            enable_action_interpolation: Whether to interpolate the current action
                with the previous action in order to produce smoother motions
            enable_action_filter: Boolean specifying if a lowpass filter should be
                used to smooth actions.  
        
        """

        # Save Parameters
        self.num_motors = num_motors
        self.num_legs = self.num_motors // dofs_per_leg
        self._pybullet_client = pybullet_client
        self._action_repeat = action_repeat
        self._self_collision_enabled = self_collision_enabled
        self._on_rack = on_rack
        self._reset_at_current_position = reset_at_current_position
        self._is_safe = True # used for termination
        
        # Motor Parameters
        self._motor_direction = motor_direction
        self._motor_offset = motor_offset
        self._observed_motor_torques = np.zeros(self.num_motors)
        self._applied_motor_torques = np.zeros(self.num_motors)
        self._max_force = 3.5
        self._pd_latency = pd_latency
        self._control_latency = control_latency
        self._motor_overheat_protection = motor_overheat_protection

        # Observations
        self._observation_noise_stdev = observation_noise_stdev
        self._observation_history = collections.deque(maxlen=100)
        self._control_observation = []

        # Link Ids
        self._chassis_link_ids = [-1]
        self._leg_link_ids = []
        self._motor_link_ids = []
        self._foot_link_ids = []

        # Sensors
        self.SetAllSensors(sensors if sensors is not None else list())

        # Actions
        self._enable_action_interpolation = enable_action_interpolation
        self._enable_action_filter = enable_action_filter
        self._last_action = None
        self._dead_zone = dead_zone

        if self._on_rack and self._reset_at_current_position:
            raise ValueError('on_rack and reset_at_current_position \
                            cannot be enabled together.')
        
        #FIX add motor instance checks

        # Simulation time step
        self.time_step = time_step

        # Episode step counter
        self._step_counter = 0 


    def get_robot(self):
        return self._robot
    
    def set_robot(self, robot):
        self._robot = robot


