import os
import inspect
import time

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from envs import tiny_robot_gym_env
from envs import tiny_robot_gym_config
from envs.env_wrappers import simple_openloop
from envs.env_wrappers import observation_dictionary_to_array_wrapper
from envs.env_wrappers import trajectory_generator_wrapper_env
from envs.env_wrappers import imitation_wrapper_env
from envs.env_wrappers import imitation_task
from envs.sensors import robot_sensors
from envs.sensors import sensor_wrappers
from envs.sensors import environment_sensors
from robots import bittle
from robots import motor_control_config

def build_imitation_env(robot_name,
                        robot_class,
                        motion_files,
                        enable_randomizer,
                        enable_rendering, 
                        include_future_frames=False):

    assert len(motion_files) > 0

    # Trajectory generator to Offset Action by Initial Pose of .52 and Set Limit of Gym Action Space
    # This is used by the trajectory generator wrapper below
    trajectory_generator = simple_openloop.TinyRobotPoseOffsetGenerator(action_limit=robot_name.UPPER_BOUND)

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

    #Calculates reward base on the difference of pose (angle of each joint), velocity (velocity of each joint), end effector (base position and base rotation), root pose, and root velocity of sim bittle and reference motion bittle
    #task = imitation_task.ImitationTask(ref_motion_filenames=motion_files,
                                        # enable_cycle_sync=True,
                                        # tar_frame_steps=[1, 2, 10, 30],
                                        # ref_state_init_prob=0.9,
                                        # warmup_time=0.25)
    task = None

    #Domain Randomization
    randomizers = []
    if enable_randomizer:
        #FIX
        pass
    
    # Initialize Open AI Gym Environment
    env = tiny_robot_gym_env.TinyRobotGymEnv(gym_config=gym_config,
                                             robot_class=robot_class,
                                             sensors=sensors,
                                             task=task, 
                                             env_randomizers=randomizers)

    #Flattens observations of individual sensors into 1 array of length 60 instead of a Box Observational Space Dictionary
    env = observation_dictionary_to_array_wrapper.ObservationDictionaryToArrayWrapper(env)

    # Apply the trajectory generator (which applies the offset) to each action predicted from the NN
    env = trajectory_generator_wrapper_env.TrajectoryGeneratorWrapperEnv(env, 
                                                                         trajectory_generator=trajectory_generator)
    curriculum_episode_length_start = 20
    curriculum_episode_length_end = 500

    # Wrapper to include future frames from imitation motion
    env = imitation_wrapper_env.ImitationWrapperEnv(env, 
                                                    episode_length_start=curriculum_episode_length_start, 
                                                    episode_length_end=curriculum_episode_length_end, 
                                                    curriculum_steps=30000000, 
                                                    include_future_frames=include_future_frames)
    return env


