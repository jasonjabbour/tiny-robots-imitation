import argparse

from robots.tiny_robot import TinyRobot
from envs import env_builder

def main():
    #Input commands
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--robot", dest="robot", type=str, default="bittle")
    args = arg_parser.parse_args()

    #Initialize robot
    robot = TinyRobot(args.robot)

    #Create Open AI Gym Environment
    env = env_builder.build_imitation_env(robot=robot)

if __name__ == '__main__':
    main()