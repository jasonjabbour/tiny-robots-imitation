from envs import tiny_robot_gym_env


def build_imitation_env(robot):


    #Initialize Open AI Gym Environment
    env = tiny_robot_gym_env.TinyRobotGymEnv(robot=robot)