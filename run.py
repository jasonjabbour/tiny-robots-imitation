import argparse
import time

from stable_baselines3 import PPO

from robots.tiny_robot import TinyRobot
from robots import calibrate_robot
from robots import bittle
from envs import env_builder

ENABLE_ENV_RANDOMIZER = True
NUM_ENV = 8
POLICY_NUM = 2

def build_model(env, output_dir):
    '''Initialize PPO Stable-baselines 3 Model'''

    model = PPO(policy='MlpPolicy', 
                env=env, 
                verbose=1, #Output Info during training
                gamma=.95, #Discount Factor
                n_epochs = 1, #Number of epoch when optimizing surrogate loss
                tensorboard_log=output_dir+'/AllPolicy'+str(POLICY_NUM)+'/log')

    return model 


def train(env, model, total_timesteps, output_dir):
    model.learn(total_timesteps=total_timesteps)
    model.save(output_dir+'/AllPolicy'+str(POLICY_NUM)+'/policy'+str(POLICY_NUM))
    env.close()

def main():
    #Input commands
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--mode', dest='mode', type=str, default='train')
    arg_parser.add_argument('--robot', dest='robot', type=str, default='bittle')
    arg_parser.add_argument('--visualize', dest='visualize', action='store_true', default=False)
    arg_parser.add_argument('--total_timesteps', dest='total_timesteps', type=int, default=6e8)
    arg_parser.add_argument('--motion_file', dest='motion_file', type=str, default='data/motions/pace_bittle.txt')
    arg_parser.add_argument('--output_dir', dest='output_dir', type=str, default='data/policies')

    args = arg_parser.parse_args()

    #Initialize robot
    if args.robot == 'bittle':
        robot_name = bittle
        robot_class = bittle.Bittle
    else:
        assert False, "Unsupported robot: " + args.robot

    #Domain Randomizer
    enable_env_rand = ENABLE_ENV_RANDOMIZER and (args.mode == 'train')

    if args.mode == 'train':

        #Create Open AI Gym Environment
        env = env_builder.build_imitation_env(robot_name=robot_name, 
                                              robot_class=robot_class,
                                              motion_files=[args.motion_file],
                                              enable_randomizer=enable_env_rand, 
                                              enable_rendering=args.visualize)

        #Build Model
        model = build_model(env=env, 
                            output_dir=args.output_dir)

        #Start RL Training Process
        train(env=env, model=model,
              total_timesteps=args.total_timesteps, 
              output_dir=args.output_dir)

    elif args.mode == 'calibrate':
        calibrate_robot.calibrate_robot(robot_name)

if __name__ == '__main__':
    main()