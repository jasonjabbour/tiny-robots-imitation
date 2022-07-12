import os
import inspect
import time

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import numpy as np
import math
import re

from robots import tiny_robot
from robots import bittle_pose_utils
from envs import tiny_robot_gym_config

NUM_MOTORS = 8
NUM_LEGS = 4
MOTOR_NAMES = [
    'FR_upper_leg_2_hip_motor_joint',
    'FR_lower_leg_2_upper_leg_joint',
    'FL_upper_leg_2_hip_motor_joint',
    'FL_lower_leg_2_upper_leg_joint',
    'RR_upper_leg_2_hip_motor_joint',
    'RR_lower_leg_2_upper_leg_joint',
    'RL_upper_leg_2_hip_motor_joint',
    'RL_lower_leg_2_upper_leg_joint'
]

INIT_RACK_POSITION = [0, 0, 1]
INIT_POSITION = [0, 0, 0.90]
JOINT_DIRECTIONS = np.array([1, 1, 1, 1, 1, 1, 1, 1])
HIP_JOINT_OFFSET = 0.0
UPPER_LEG_JOINT_OFFSET = 0.0
KNEE_JOINT_OFFSET = 0.0
DOFS_PER_LEG = 2 #one for each servo motor (rotational DF)
JOINT_OFFSETS = np.array(
    [UPPER_LEG_JOINT_OFFSET, KNEE_JOINT_OFFSET] * 4)
PI = math.pi

#FIX
MAX_MOTOR_ANGLE_CHANGE_PER_STEP = 0.2
#FIX
_DEFAULT_HIP_POSITIONS = (
    (0.21, -0.1157, 0),
    (0.21, 0.1157, 0),
    (-0.21, -0.1157, 0),
    (-0.21, 0.1157, 0),
)

#FIX
HIP_P_GAIN = 2.75
HIP_D_GAIN = .025
KNEE_P_GAIN = 2.75
KNEE_D_GAIN = .025

# Initial Motor Angles
INIT_MOTOR_ANGLES = np.array([
    bittle_pose_utils.BITTLE_DEFAULT_HIP_ANGLE,
    bittle_pose_utils.BITTLE_DEFAULT_KNEE_ANGLE
] * NUM_LEGS)

#Identify Joints from URDF File
_CHASSIS_NAME_PATTERN = re.compile(r'\w+_chassis_\w+')
_MOTOR_NAME_PATTERN = re.compile(r'\w+_hip_motor_\w+')
_KNEE_NAME_PATTERN = re.compile(r'\w+_lower_leg_\w+')
_TOE_NAME_PATTERN = re.compile(r'jtoe\d*')

URDF_FILENAME = 'models/bittle.urdf'

# _BODY_B_FIELD_NUMBER = 2
# _LINK_A_FIELD_NUMBER = 3

#NEW
UPPER_BOUND = 1
LOWER_BOUND = -1

class Bittle(tiny_robot.TinyRobot):
    """A simulation for the Bittle robot."""
    #CHANGE these values (not used)
    # MPC_BODY_MASS = 2.64/9.8 #.27kg
    # MPC_BODY_INERTIA = (0, 0, 0, 0, 0, 0, 0, 0)
    # MPC_BODY_HEIGHT = 0.42 
    ACTION_CONFIG = [
      tiny_robot_gym_config.ScalarField(name="FR_upper_leg_2_hip_motor_joint",
                                        upper_bound=UPPER_BOUND,
                                        lower_bound=LOWER_BOUND),
      tiny_robot_gym_config.ScalarField(name="FR_lower_leg_2_upper_leg_joint",
                                        upper_bound=UPPER_BOUND,
                                        lower_bound=LOWER_BOUND),
      tiny_robot_gym_config.ScalarField(name="FL_upper_leg_2_hip_motor_joint",
                                        upper_bound=UPPER_BOUND,
                                        lower_bound=LOWER_BOUND),
      tiny_robot_gym_config.ScalarField(name="FL_lower_leg_2_upper_leg_joint",
                                        upper_bound=UPPER_BOUND,
                                        lower_bound=LOWER_BOUND),
      tiny_robot_gym_config.ScalarField(name="RR_upper_leg_2_hip_motor_joint",
                                        upper_bound=UPPER_BOUND,
                                        lower_bound=LOWER_BOUND),
      tiny_robot_gym_config.ScalarField(name="RR_lower_leg_2_upper_leg_joint",
                                        upper_bound=UPPER_BOUND,
                                        lower_bound=LOWER_BOUND),
      tiny_robot_gym_config.ScalarField(name="RL_upper_leg_2_hip_motor_joint",
                                        upper_bound=UPPER_BOUND,
                                        lower_bound=LOWER_BOUND),
      tiny_robot_gym_config.ScalarField(name="RL_lower_leg_2_upper_leg_joint",
                                        upper_bound=UPPER_BOUND,
                                        lower_bound=LOWER_BOUND)
    ]

    def __init__(
            self,
            pybullet_client, 
            motor_control_mode, 
            urdf_filename=URDF_FILENAME, 
            enable_clip_motor_commands=False, 
            time_step=0.001,
            action_repeat=33, 
            sensors=None, 
            control_latency=0.002, 
            on_rack=False, 
            enable_action_interpolation=False, 
            enable_action_filter=False, 
            reset_time=-1, 
            allow_knee_contact=False, 
            dead_zone=False):
        
        self._urdf_filename = urdf_filename
        self._allow_knee_contact = allow_knee_contact
        self._enable_clip_motor_commands = enable_clip_motor_commands

        #FIX PD controller
        motor_kp = None
        motor_kd = None

        # Initialize Super Tiny Robot Class
        super(Bittle, self).__init__(
            pybullet_client=pybullet_client, 
            time_step=time_step, 
            action_repeat=action_repeat, 
            num_motors=NUM_MOTORS, 
            dofs_per_leg=DOFS_PER_LEG, 
            motor_direction=JOINT_DIRECTIONS, 
            motor_offset=JOINT_OFFSETS, 
            motor_overheat_protection=False, 
            motor_control_mode=motor_control_mode, 
            motor_model_class=None, #FIX
            sensors=sensors, 
            motor_kp=motor_kp, #FIX
            motor_kd=motor_kd, #FIX
            control_latency=control_latency, 
            on_rack=on_rack, 
            enable_action_interpolation=enable_action_interpolation, 
            enable_action_filter=enable_action_filter, 
            reset_time=reset_time, 
            dead_zone=dead_zone)

    def _LoadRobotURDF(self):
        """Load Robot using URDF File"""
        # bittle_urdf_path = self.GetURDFFile()

        # if self._self_collision_enabled:
        #     self.quadruped = self._pybullet_client.loadURDF(
        #         bittle_urdf_path, 
        #     )

        pass
    

    def GetURDFFile(self):
        return self._urdf_filename