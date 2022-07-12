import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import attr
from gym import spaces
import numpy as np

from robots import bittle_pose_utils

class TinyRobotPoseOffsetGenerator(object):
    '''A generator that returns the offset motor angles'''
    def __init__(
        self, 
        init_hip=bittle_pose_utils.BITTLE_DEFAULT_HIP_ANGLE,
        init_knee=bittle_pose_utils.BITTLE_DEFAULT_KNEE_ANGLE,
        action_limit=0.5
    ):
        '''Initializes the NN controller.'''

        self._pose=np.array(
            attr.astuple(
                bittle_pose_utils.BittlePose(hip_angle_0=init_hip,
                                                    knee_angle_0=init_knee,
                                                    hip_angle_1=init_hip,
                                                    knee_angle_1=init_knee,
                                                    hip_angle_2=init_hip,
                                                    knee_angle_2=init_knee,
                                                    hip_angle_3=init_hip,
                                                    knee_angle_3=init_knee)))
            
        # Set action limit for all 8 joints 
        action_high = np.array([action_limit]*8)

        #Declare the OpenAI Gym Action Space with action limit
        self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)

    def reset(self):
        pass

    def get_action(self, current_time=None, input_action=None):
        '''Computes the trajectory according to input time and action
        Args:
            current_time: The time in gym env since reset.
            input_action: A numpy array. The input leg pose from a NN controller.

        Returns:
            A numpy array. The desired motor angles            
        '''
        del current_time
        return self._pose + input_action

    def get_observation(self, input_observation):
        '''Get the trajectory generator's observation'''

        return input_observation

if __name__ == '__main__':
    offset_generator = TinyRobotPoseOffsetGenerator()

    # Initial Pose
    print(offset_generator._pose)

    #Wrapper takes input action and offsets by initial pose
    print(offset_generator.get_action(input_action=[.1]*8))

