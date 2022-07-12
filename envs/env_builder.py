import os
import inspect
import time

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from envs import tiny_robot_gym_env
from envs import tiny_robot_gym_config
from envs.env_wrappers import simple_openloop
from envs.sensors import robot_sensors
from envs.sensors import sensor_wrappers
from envs.sensors import environment_sensors
from robots import bittle
from robots import motor_control_config

def build_imitation_env(robot_name,
                        robot_class,
                        motion_files,
                        enable_randomizer,
                        enable_rendering):

    assert len(motion_files) > 0

    # Wrapper to Offset Action by Initial Pose of .52 and Set Limit of Gym Action Space
    trajectory_generator = simple_openloop.TinyRobotPoseOffsetGenerator(action_limit=robot_name.UPPER_BOUND)

    # FIX
    curriculum_episode_length_start = 20
    curriculum_episode_length_end = 500

    # Customize Simulation Parameters
    sim_params = tiny_robot_gym_config.SimulationParameters()
    sim_params.enable_rendering = enable_rendering
    sim_params.allow_knee_contact = True
    sim_params.motor_control_mode = motor_control_config.MotorControlMode.POSITION

    # Group the Simulation Parameters into the TinyRobotGymConfig Class
    gym_config = tiny_robot_gym_config.TinyRobotGymConfig(simulation_parameters=sim_params)

    # Create a list of sensors on the robot.
    # First initialize the sensor then pass that into 
    # a history wrapper to keep track of the sensor's history
    sensors = [
        # IMU Sensor [Roll, Pitch, Yaw] * 6 = 18
        sensor_wrappers.HistoricSensorWrapper(wrapped_sensor=robot_sensors.IMUSensor(), num_history=6), 
        # Last Action Sensor 8 Motors * 3 = 24
        sensor_wrappers.HistoricSensorWrapper(wrapped_sensor=environment_sensors.LastActionSensor(num_actions=robot_name.NUM_MOTORS), num_history=3)
    ]

    # Create the reward function
    # Calculates reward base on the difference of pose (angle of each joint),
    # velocity (velocity of each joint), end effector (base position and base rotation), 
    # root pose, and root velocity of sim bittle and reference motion bittle

    #FIX

    #Domain Randomization
    randomizers = []
    if enable_randomizer:
        #FIX
        pass

    # Initialize Open AI Gym Environment
    env = tiny_robot_gym_env.TinyRobotGymEnv(gym_config=gym_config,
                                             robot_class=robot_class,
                                             sensors=sensors,
                                             task=None, 
                                             env_randomizers=randomizers)

    print("MADE IT")


