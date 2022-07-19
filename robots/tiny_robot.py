"""This file implements the functionalities of a Tiny Toboy using pybullet"""

import os
import inspect
from turtle import onrelease

# MOVE
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

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import collections
import copy
import math
import re
import numpy as np
from datetime import datetime

from robots import math_conversion_utils
from robots import motor_control_config

_CHASSIS_NAME_PATTERN = re.compile(r"chassis\D*center")
_MOTOR_NAME_PATTERN = re.compile(r"motor\D*joint")
_KNEE_NAME_PATTERN = re.compile(r"knee\D*")
_BRACKET_NAME_PATTERN = re.compile(r"motor\D*_bracket_joint")
_LEG_NAME_PATTERN1 = re.compile(r"hip\D*joint")
_LEG_NAME_PATTERN2 = re.compile(r"hip\D*link")
_LEG_NAME_PATTERN3 = re.compile(r"motor\D*link")

LEG_POSITION = ["front_left", "back_left", "front_right", "back_right"]
KNEE_CONSTRAINT_POINT_RIGHT = [0, 0.005, 0.2] # FIX
KNEE_CONSTRAINT_POINT_LEFT = [0, 0.01, 0.2] # FIX

SENSOR_NOISE_STDDEV = (0.0, 0.0, 0.0, 0.1, 0.0) #NEW was all zeroz
DEFAULT_MOTOR_DIRECTIONS = (1, 1, 1, 1, 1, 1, 1, 1) # use -1 to denote oposite direction
DEFAULT_MOTOR_OFFSETS = (0, 0, 0, 0, 0, 0, 0, 0)

INIT_POSITION = [0, 0, .2]
INIT_RACK_POSITION = [0, 0, 1]
INIT_ORIENTATION = [0, 0, 0, 1]

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
        self._max_velocity = .07
        self._pd_latency = pd_latency
        self._control_latency = control_latency
        self._motor_overheat_protection = motor_overheat_protection
        self._motor_control_mode = motor_control_mode

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

        # This also includes time spend during Reset motion.
        self._state_action_counter = 0 

        # Invert initial orientation
        _, self._init_orientation_inv = self._pybullet_client.invertTransform(
            position=[0,0,0], orientation=self._GetDefaultInitOrientation())

        # Action filter
        if self._enable_action_filter:
            pass

        # Reset. (-1.0 means skipping the reset motion)
        self.Reset(reset_time=reset_time)
        
        # Get Observation
        self.ReceiveObservation()

    def GetTimeSinceReset(self):
        """Calculate the time in simulation using steps and time per step"""
        return self._step_counter * self.time_step 
    
    def Step(self, action, control_mode=None):
        """Step simulation"""

        # Initialize Action Filter if exists
        if self._enable_action_filter:
            pass

        # Set motor control mode
        if control_mode==None:
            control_mode = self._motor_control_mode
    
        # Create Dead-band zone. 
        if self._dead_zone == True and self._last_action is not None:
            # Check each action
            for joint_num, angle in enumerate(action):
                # Motor is not moved if action is less than 1 deg change. 
                if abs(angle-self._last_action[joint_num]) <= .0175: #1deg
                    action[joint_num] = self._last_action[joint_num]
        
        # Interpolation used to control motors if allowed
        for i in range(self._action_repeat): 
            # If enabled interpolation action action will be processed
            proc_action = self.ProcessAction(action, i)
            # Apply action to robot
            self._StepInternal(proc_action, control_mode)
            self._step_counter += 1
        self._last_action = action
    
        return proc_action
    
    def ProcessAction(self, action, substep_count):
        """If enabled, interpolates between the current and previous actions.

        Args:
        action: current action.
        substep_count: the step count should be between [0, self.__action_repeat).

        Returns:
        If interpolation is enabled, returns interpolated action depending on
        the current action repeat substep.
        """
        if self._enable_action_interpolation and self._last_action is not None:
            lerp = float(substep_count + 1) / self._action_repeat
            proc_action = self._last_action + lerp * (action - self._last_action)
        else:
            proc_action = action

        return proc_action
 
    
    def _StepInternal(self, action, motor_control_mode):
        """Apply action to robot and step the simulation"""
        self.ApplyAction(action, motor_control_mode)
        #FIX change the time step here
        # 10 HZ
        for i in range(24):
            self._pybullet_client.stepSimulation()
        self.ReceiveObservation()
        self._state_action_counter+=1 
    
    def ApplyAction(self, motor_commands, motor_control_mode):
        """ Apply the motor commands using the motor model.

        Args:
            motor_commands: np.array. Can be motor angles, torques, hybrid commands, 
                or motor pwms. 
            motor_control_mode: A MotorControlMode enum.      
        
        """
        self.last_action_time = self._state_action_counter * self.time_step
        control_mode = motor_control_mode

        # Choose class motor control mode if none provided
        if control_mode is None:
            control_mode = self._motor_control_mode  

        # Convert actions to np array
        motor_commands = np.asarray(motor_commands) 

        # REMOVED delayed observations motor velocities observations
        # REMOVED Motor overheat protection
        # REMOVED SetMotorTorques of to move each joint

        motor_ids = []

        # Choose the motors that are enabled
        for motor_id, motor_enabled in zip(self._motor_id_list, self._motor_enabled_list):
            if motor_enabled:
                motor_ids.append(motor_id)

        self._SetDesiredMotorAngleByID(motor_ids, motor_commands)

    def _SetDesiredMotorAngleByID(self, motor_ids, positions):
        '''Using Position control attempt to move each joint to target position, 
        
        Args:
            motor_ids: ids of motors from URDF file, 
            position: target position in radians for each joint
        
        '''
        self._pybullet_client.setJointMotorControlArray(
            bodyIndex=self.quadruped, 
            jointIndices=motor_ids, 
            controlMode=self._pybullet_client.POSITION_CONTROL,
            targetPositions=positions,
            forces=[.5]*self.num_motors,
        ) #FIX maxvelocity

    def Terminate(self):
        pass

    def GetFootLinkIDs(self):
        '''Get list of IDS for all foot links'''
        return self._foot_link_ids
    
    def _RecordMassInfoFromURDF(self):
        '''Records the mass information from the URDF file.'''

        #Get the mass of the links
        self._base_mass_urdf = []
        for chassis_id in self._chassis_link_ids:
            self._base_mass_urdf.append(
                self._pybullet_client.getDynamicsInfo(self.quadruped, chassis_id)[0])
        self._leg_masses_urdf = []
        for leg_id in self._leg_link_ids:
            self._leg_masses_urdf.append(
                self._pybullet_client.getDynamicsInfo(self.quadruped, leg_id)[0])
        for motor_id in self._motor_link_ids: 
            self._leg_masses_urdf.append(
                self._pybullet_client.getDynamicsInfo(self.quadruped, motor_id)[0])
    
    def _RecordInertiaInfoFromURDF(self):
        '''Record the inertia of each body from URDF file.'''
        self._link_urdf = []
        num_bodies = self._pybullet_client.getNumJoints(self.quadruped)

        # Get the inertia of every body in URDF
        for body_id in range(-1, num_bodies): #-1 is for the base link
            inertia = self._pybullet_client.getDynamicsInfo(self.quadruped, body_id)[2]
            self._link_urdf.append(inertia)

        # Assign the inertia for base, legs, and motors
        # We need to use id+1 to index self._link_urdf because it has the base
        # (index = -1) at the first element.
        self._base_inertia_urdf = [
            self._link_urdf[chassis_id + 1]
            for chassis_id in self._chassis_link_ids
        ]
        self._leg_inertia_urdf = [
            self._link_urdf[leg_id + 1] for leg_id in self._leg_link_ids
        ]
        self._leg_inertia_urdf.extend(
            [self._link_urdf[motor_id + 1] for motor_id in self._motor_link_ids])

    def _BuildUrdfIds(self):
        '''Build the link Ids from its name in the URDF file.
        
        Raises:
            ValueError: Unknown category of the join name

        Implemented in subclass
        '''
        pass

    def _BuildJointNameToIdDict(self):
        '''Get all joints from urdf and create a dictionary.
        
        Example:
            {'battery-joint': 0, 'cover-joint': 1, 'RL_upper_leg_2_hip_motor_joint': 2,
            'RL_lower_leg_2_upper_leg_joint': 3, 'FL_upper_leg_2_hip_motor_joint': 4,
            'FL_lower_leg_2_upper_leg_joint': 5, 'mainboard_joint': 6, 'imu_joint': 7,
            'RR_upper_leg_2_hip_motor_joint': 8, 'RR_lower_leg_2_upper_leg_joint': 9,
            'FR_upper_leg_2_hip_motor_joint': 10, 'FR_lower_leg_2_upper_leg_joint': 11}
        '''
        num_joints = self._pybullet_client.getNumJoints(self.quadruped)
        self._joint_name_to_id = {}
        for i in range(num_joints):
            joint_info = self._pybullet_client.getJointInfo(self.quadruped, i)
            self._joint_name_to_id[joint_info[1].decode("UTF-8")] = joint_info[0]

    def _BuildMotorIdList(self):
        '''Create list of the ids for each motor joint
        
        Example:
            [10, 11, 4, 5, 8, 9, 2, 3]
        '''
        self._motor_id_list = [
            self._joint_name_to_id[motor_name]
            for motor_name in self._GetMotorNames()
        ]

    def IsObservationValid(self):
        """Whether the observation is valid for the current time step.

        In simulation, observations are always valid. In real hardware, it may not
        be valid from time to time when communication error happens between the
        Nvidia TX2 and the microcontroller.

        Returns:
        Whether the observation is valid for the current time step.
        """
        return True

    def _RemoveDefaultJointDamping(self):
        '''Set the linear and angular damping to 0'''
        num_joints = self._pybullet_client.getNumJoints(self.quadruped)
        for i in range(num_joints):
            # Get all the joint information from URDF
            joint_info = self._pybullet_client.getJointInfo(self.quadruped, i)
            # For the joint set the linear and angular damping forces to zero
            self._pybullet_client.changeDynamics(joint_info[0],
                                                -1,
                                                linearDamping=0,
                                                angularDamping=0)
    

    def Reset(self, reload_urdf=True, default_motor_angles=None, reset_time=3.0):
        """Reset the minitaur to its initial states.

        Args:
        reload_urdf: Whether to reload the urdf file. If not, Reset() just place
            the minitaur back to its starting position.
        default_motor_angles: The default motor angles. If it is None, minitaur
            will hold a default pose (motor angle math.pi / 2) for 100 steps. In
            torque control mode, the phase of holding the default pose is skipped.
        reset_time: The duration (in seconds) to hold the default motor angles. If
            reset_time <= 0 or in torque control mode, the phase of holding the
            default pose is skipped.
        """

        if reload_urdf:
            self._LoadRobotURDF()

            self._BuildJointNameToIdDict()
            self._BuildUrdfIds()
            self._RemoveDefaultJointDamping()
            self._BuildMotorIdList()
            self._RecordMassInfoFromURDF()
            self._RecordInertiaInfoFromURDF()
            self.ResetPose(add_constraint=True)
        else:
            self._pybullet_client.resetBasePositionAndOrientation(
                self.quadruped, self._GetDefaultInitPosition(),
                self._GetDefaultInitOrientation())
            self._pybullet_client.resetBaseVelocity(self.quadruped, [0, 0, 0],
                                                    [0, 0, 0])
            self.ResetPose(add_constraint=False)

        self._motor_enabled_list = [True] * self.num_motors
        self._observation_history.clear()
        self._step_counter = 0
        self._state_action_counter = 0
        self._is_safe = True
        self._last_action = None
        self._SettleDownForReset(default_motor_angles, reset_time)

        if self._enable_action_filter:
            pass
    
    def _SettleDownForReset(self, default_motor_angles, reset_time):
        """Sets the default motor angles and waits for the robot to settle down.

        The reset is skipped is reset_time is less than zereo.

        Args:
        default_motor_angles: A list of motor angles that the robot will achieve
            at the end of the reset phase.
        reset_time: The time duration for the reset phase.
        """
        if reset_time <= 0:
          return
        
        # FIX
        return
    
    def ResetPose(self, add_constraint):
        """Reset the pose of the minitaur.

        Args:
        add_constraint: Whether to add a constraint at the joints of two feet.
        """
        for i in range(self.num_legs):
            self._ResetPoseForLeg(i, add_constraint)

    def _ResetPoseForLeg(self, leg_id, add_constraint):
        """Reset the initial pose for the leg.

        Args:
        leg_id: It should be 0, 1, 2, or 3, which represents the leg at
            front_left, back_left, front_right and back_right.
        add_constraint: Whether to add a constraint at the joints of two feet.
        """
        knee_friction_force = 0
        half_pi = math.pi / 2.0
        knee_angle = -2.1834

        leg_position = LEG_POSITION[leg_id]
        self._pybullet_client.resetJointState(
            self.quadruped,
            self._joint_name_to_id["motor_" + leg_position + "L_joint"],
            self._motor_direction[2 * leg_id] * half_pi,
            targetVelocity=0)
        self._pybullet_client.resetJointState(
            self.quadruped,
            self._joint_name_to_id["knee_" + leg_position + "L_link"],
            self._motor_direction[2 * leg_id] * knee_angle,
            targetVelocity=0)
        self._pybullet_client.resetJointState(
            self.quadruped,
            self._joint_name_to_id["motor_" + leg_position + "R_joint"],
            self._motor_direction[2 * leg_id + 1] * half_pi,
            targetVelocity=0)
        self._pybullet_client.resetJointState(
            self.quadruped,
            self._joint_name_to_id["knee_" + leg_position + "R_link"],
            self._motor_direction[2 * leg_id + 1] * knee_angle,
            targetVelocity=0)
        # Constraints connect links. Better for solving physics together rather than individually
        if add_constraint:
            self._pybullet_client.createConstraint(
                self.quadruped,
                self._joint_name_to_id["knee_" + leg_position + "R_link"],
                self.quadruped,
                self._joint_name_to_id["knee_" + leg_position + "L_link"],
                self._pybullet_client.JOINT_POINT2POINT, [0, 0, 0],
                KNEE_CONSTRAINT_POINT_RIGHT, KNEE_CONSTRAINT_POINT_LEFT)

        # Disable the default motor in pybullet.
        self._pybullet_client.setJointMotorControl2(
            bodyIndex=self.quadruped,
            jointIndex=(self._joint_name_to_id["motor_" + leg_position +
                                            "L_joint"]),
            controlMode=self._pybullet_client.VELOCITY_CONTROL,
            targetVelocity=0,
            force=knee_friction_force)
        self._pybullet_client.setJointMotorControl2(
            bodyIndex=self.quadruped,
            jointIndex=(self._joint_name_to_id["motor_" + leg_position +
                                            "R_joint"]),
            controlMode=self._pybullet_client.VELOCITY_CONTROL,
            targetVelocity=0,
            force=knee_friction_force)
        self._pybullet_client.setJointMotorControl2(
            bodyIndex=self.quadruped,
            jointIndex=(self._joint_name_to_id["knee_" + leg_position + "L_link"]),
            controlMode=self._pybullet_client.VELOCITY_CONTROL,
            targetVelocity=0,
            force=knee_friction_force)
        self._pybullet_client.setJointMotorControl2(
            bodyIndex=self.quadruped,
            jointIndex=(self._joint_name_to_id["knee_" + leg_position + "R_link"]),
            controlMode=self._pybullet_client.VELOCITY_CONTROL,
            targetVelocity=0,
            force=knee_friction_force)

    def _LoadRobotURDF(self):
        """Loads the URDF file for the robot.
        
        Depending on class settings, either enable self collision or not.
        
        """
        urdf_file = self.GetURDFFile()
        if self._self_collision_enabled:
            self.quadruped = self._pybullet_client.loadURDF(
                urdf_file,
                self._GetDefaultInitPosition(),
                self._GetDefaultInitOrientation(),
                flags=self._pybullet_client.URDF_USE_SELF_COLLISION)
        else:
            self.quadruped = self._pybullet_client.loadURDF(
                urdf_file, self._GetDefaultInitPosition(),
                self._GetDefaultInitOrientation())
    
    def GetURDFFile(self):
        return 'models/bittle.urdf'

    def GetBasePosition(self):
        """Get the position of minitaur's base.

        Returns:
        The position of minitaur's base.
        """
        return self._base_position

    def GetBaseVelocity(self):
        """Get the linear velocity of minitaur's base.

        Returns:
        The velocity of minitaur's base.
        """
        velocity, _ = self._pybullet_client.getBaseVelocity(self.quadruped)
        return velocity

    def GetTrueBaseRollPitchYaw(self):
        """Get minitaur's base orientation in euler angle in the world frame.

        Returns:
        A tuple (roll, pitch, yaw) of the base in world frame.
        """
        orientation = self.GetTrueBaseOrientation()
        roll_pitch_yaw = self._pybullet_client.getEulerFromQuaternion(orientation)
        return np.asarray(roll_pitch_yaw)

    def GetBaseRollPitchYaw(self):
        """Get minitaur's base orientation in euler angle in the world frame.

        This function mimicks the noisy sensor reading and adds latency.
        Returns:
        A tuple (roll, pitch, yaw) of the base in world frame polluted by noise
        and latency.
        """

        # REMOVED Delayed observations
        orientation = self.GetTrueBaseOrientation()

        roll_pitch_yaw = self._pybullet_client.getEulerFromQuaternion(
            orientation)
        noisy_roll_pitch_yaw = self._AddSensorNoise(np.array(roll_pitch_yaw),
                                            self._observation_noise_stdev[3])
        return noisy_roll_pitch_yaw

    def GetTrueBaseOrientation(self):
        """Get the orientation of robot's base, represented as quaternion.

        Returns:
        The orientation of robot's base.
        """
        return self._base_orientation

    def GetBaseOrientation(self):
        """Get the orientation of robot's base, represented as quaternion.

        This function mimicks the noisy sensor reading.
        Returns:
        The orientation of minitaur's base polluted by noise.
        """
        return self._pybullet_client.getQuaternionFromEuler(
            self.GetBaseRollPitchYaw())
        

    def GetActionDimension(self):
        """Get the length of the action list.

        Returns:
        The length of the action list.
        """
        return self.num_motors

    def GetBaseMassesFromURDF(self):
        """Get the mass of the base from the URDF file."""
        return self._base_mass_urdf

    def GetBaseInertiasFromURDF(self):
        """Get the inertia of the base from the URDF file."""
        return self._base_inertia_urdf

    def GetLegMassesFromURDF(self):
        """Get the mass of the legs from the URDF file."""
        return self._leg_masses_urdf

    def GetLegInertiasFromURDF(self):
        """Get the inertia of the legs from the URDF file."""
        return self._leg_inertia_urdf

    def SetBaseMasses(self, base_mass):
        """Set the mass of minitaur's base.

        Args:
        base_mass: A list of masses of each body link in CHASIS_LINK_IDS. The
            length of this list should be the same as the length of CHASIS_LINK_IDS.

        Raises:
        ValueError: It is raised when the length of base_mass is not the same as
            the length of self._chassis_link_ids.
        """
        if len(base_mass) != len(self._chassis_link_ids):
            raise ValueError(
                "The length of base_mass {} and self._chassis_link_ids {} are not "
                "the same.".format(len(base_mass), len(self._chassis_link_ids)))
        for chassis_id, chassis_mass in zip(self._chassis_link_ids, base_mass):
            self._pybullet_client.changeDynamics(self.quadruped,
                                            chassis_id,
                                            mass=chassis_mass)


    def SetLegMasses(self, leg_masses):
        """Set the mass of the legs.

        A leg includes leg_link and motor. 4 legs contain 16 links (4 links each)
        and 8 motors. First 16 numbers correspond to link masses, last 8 correspond
        to motor masses (24 total).

        Args:
        leg_masses: The leg and motor masses for all the leg links and motors.

        Raises:
        ValueError: It is raised when the length of masses is not equal to number
            of links + motors.
        """
        if len(leg_masses) != len(self._leg_link_ids) + len(self._motor_link_ids):
            raise ValueError("The number of values passed to SetLegMasses are "
                        "different than number of leg links and motors.")
        for leg_id, leg_mass in zip(self._leg_link_ids, leg_masses):
            self._pybullet_client.changeDynamics(self.quadruped,
                                            leg_id,
                                            mass=leg_mass)
        motor_masses = leg_masses[len(self._leg_link_ids):]
        for link_id, motor_mass in zip(self._motor_link_ids, motor_masses):
            self._pybullet_client.changeDynamics(self.quadruped,
                                            link_id,
                                            mass=motor_mass)

    def SetBaseInertias(self, base_inertias):
        """Set the inertias of minitaur's base.

        Args:
        base_inertias: A list of inertias of each body link in CHASIS_LINK_IDS.
            The length of this list should be the same as the length of
            CHASIS_LINK_IDS.

        Raises:
        ValueError: It is raised when the length of base_inertias is not the same
            as the length of self._chassis_link_ids and base_inertias contains
            negative values.
        """
        if len(base_inertias) != len(self._chassis_link_ids):
            raise ValueError(
                "The length of base_inertias {} and self._chassis_link_ids {} are "
                "not the same.".format(len(base_inertias),
                                    len(self._chassis_link_ids)))
        for chassis_id, chassis_inertia in zip(self._chassis_link_ids,
                                            base_inertias):
            for inertia_value in chassis_inertia:
                if (np.asarray(inertia_value) < 0).any():
                    raise ValueError("Values in inertia matrix should be non-negative.")
            self._pybullet_client.changeDynamics(
                self.quadruped, chassis_id, localInertiaDiagonal=chassis_inertia)


    def SetLegInertias(self, leg_inertias):
        """Set the inertias of the legs.

        A leg includes leg_link and motor. 4 legs contain 16 links (4 links each)
        and 8 motors. First 16 numbers correspond to link inertia, last 8 correspond
        to motor inertia (24 total).

        Args:
        leg_inertias: The leg and motor inertias for all the leg links and motors.

        Raises:
        ValueError: It is raised when the length of inertias is not equal to
        the number of links + motors or leg_inertias contains negative values.
        """

        if len(leg_inertias) != len(self._leg_link_ids) + len(
            self._motor_link_ids):
            raise ValueError("The number of values passed to SetLegMasses are "
                        "different than number of leg links and motors.")
        for leg_id, leg_inertia in zip(self._leg_link_ids, leg_inertias):
            for inertia_value in leg_inertias:
                if (np.asarray(inertia_value) < 0).any():
                    raise ValueError("Values in inertia matrix should be non-negative.")
            self._pybullet_client.changeDynamics(self.quadruped,
                                            leg_id,
                                            localInertiaDiagonal=leg_inertia)

        motor_inertias = leg_inertias[len(self._leg_link_ids):]
        for link_id, motor_inertia in zip(self._motor_link_ids, motor_inertias):
            for inertia_value in motor_inertias:
                if (np.asarray(inertia_value) < 0).any():
                    raise ValueError("Values in inertia matrix should be non-negative.")
            self._pybullet_client.changeDynamics(self.quadruped,
                                            link_id,
                                            localInertiaDiagonal=motor_inertia)

    def SetFootFriction(self, foot_friction):
        """Set the lateral friction of the feet.

        Args:
        foot_friction: The lateral friction coefficient of the foot. This value is
            shared by all four feet.
        """
        for link_id in self._foot_link_ids:
            self._pybullet_client.changeDynamics(self.quadruped,
                                           link_id,
                                           lateralFriction=foot_friction)

    def GetTrueObservation(self):
        observation = []

        observation.extend(self.GetTrueBaseOrientation())
        return observation


    def ReceiveObservation(self):
        """Receive the observation from sensors.

        This function is called once per step. The observations are only updated
        when this function is called.
        """
        self._joint_states = self._pybullet_client.getJointStates(
            self.quadruped, self._motor_id_list)
        self._base_position, orientation = (
            self._pybullet_client.getBasePositionAndOrientation(self.quadruped))
        # Computes the relative orientation relative to the robot's
        # initial_orientation.
        _, self._base_orientation = self._pybullet_client.multiplyTransforms(
            positionA=[0, 0, 0],
            orientationA=orientation,
            positionB=[0, 0, 0],
            orientationB=self._init_orientation_inv)
        self._observation_history.appendleft(self.GetTrueObservation())
        self._control_observation = self._GetControlObservation()
        self.last_state_time = self._state_action_counter * self.time_step


    def _GetControlObservation(self):
        control_delayed_observation = self._GetDelayedObservation(
        self._control_latency)
        return control_delayed_observation


    def _GetDelayedObservation(self, latency):
        """Get observation that is delayed by the amount specified in latency.

    `    Args:
        latency: The latency (in seconds) of the delayed observation.

        Returns:
        observation: The observation which was actually latency seconds ago.
        """


        if latency <= 0 or len(self._observation_history) == 1:
            return self._observation_history[0]

        # REMOVED delayed observations
        return self._observation_history[0]

    def _AddSensorNoise(self, sensor_values, noise_stdev):
        if noise_stdev <= 0:
            return sensor_values
        observation = sensor_values + np.random.normal(scale=noise_stdev,
                                                    size=sensor_values.shape)
        return observation


    def SetControlLatency(self, latency):
        """Set the latency of the control loop.

        It measures the duration between sending an action from Nvidia TX2 and
        receiving the observation from microcontroller.

        Args:
        latency: The latency (in seconds) of the control loop.
        """
        self._control_latency = latency
    

    def GetControlLatency(self):
        """Get the control latency.

        Returns:
        The latency (in seconds) between when the motor command is sent and when
            the sensor measurements are reported back to the controller.
        """
        return self._control_latency

    def GetNumKneeJoints(self):
        return len(self._foot_link_ids)

    def _GetMotorNames():
        '''Return names of motor joints'''
        return MOTOR_NAMES

    def _GetDefaultInitPosition(self):
        """Returns the init position of the robot.

        It can be either 1) origin (INIT_POSITION), 2) origin with a rack
        (INIT_RACK_POSITION), or 3) the previous position.
        """
        # If we want continuous resetting and is not the first episode.
        if self._reset_at_current_position and self._observation_history:
            x, y, _ = self.GetBasePosition()
            _, _, z = INIT_POSITION
            return [x, y, z]

        if self._on_rack:
            return INIT_RACK_POSITION
        else:
            return INIT_POSITION
    
    def _GetDefaultInitOrientation(self):
        """Returns the init position of the robot.

        It can be either 1) INIT_ORIENTATION or 2) the previous rotation in yaw.
        """
        # If we want continuous resetting and is not the first episode.
        if self._reset_at_current_position and self._observation_history:
            _, _, yaw = self.GetBaseRollPitchYaw()
            return self._pybullet_client.getQuaternionFromEuler([0.0, 0.0, yaw])
        
        return 

    @property
    def chassis_link_ids(self):
        return self._chassis_link_ids

    def SetAllSensors(self, sensors):
        """set all sensors to this robot and move the ownership to this robot.

        Args:
        sensors: a list of sensors to this robot.
        """
        for s in sensors:
            s.set_robot(self)
        self._sensors = sensors

    def GetAllSensors(self):
        """get all sensors associated with this robot.

        Returns:
        sensors: a list of all sensors.
        """
        return self._sensors

    def GetSensor(self, name):
        """get the first sensor with the given name.

        This function return None if a sensor with the given name does not exist.

        Args:
        name: the name of the sensor we are looking

        Returns:
        sensor: a sensor with the given name. None if not exists.
        """
        for s in self._sensors:
            if s.get_name() == name:
                return s
        return None

    @property
    def is_safe(self):
        return self._is_safe

    @property
    def last_action(self):
        return self._last_action
    
    @property
    def pybullet_client(self):
        return self._pybullet_client
    
    @property
    def joint_states(self):
        return self._joint_states  



