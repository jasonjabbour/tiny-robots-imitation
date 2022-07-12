'''Utility functions to calculate Tiny Robot's Pose and Motor Angles.'''

import attr

BITTLE_DEFAULT_HIP_ANGLE = 0.52
BITTLE_DEFAULT_KNEE_ANGLE = 0.52

@attr.s
class BittlePose(object):
    '''Default pose of Tiny Robot
    
        Leg order:
        0 -> Front Right.
        1 -> Front Left.
        2 -> Rear Right.
        3 -> Rear Left.

                    Left

                3---------------1
           Back |               | Front --->
                2---------------0

                    Right
    '''
    hip_angle_0 = attr.ib(type=float, default=0)
    knee_angle_0 = attr.ib(type=float, default=0)
    hip_angle_1 = attr.ib(type=float, default=0)
    knee_angle_1 = attr.ib(type=float, default=0)
    hip_angle_2 = attr.ib(type=float, default=0)
    knee_angle_2 = attr.ib(type=float, default=0)
    hip_angle_3 = attr.ib(type=float, default=0)
    knee_angle_3 = attr.ib(type=float, default=0)
