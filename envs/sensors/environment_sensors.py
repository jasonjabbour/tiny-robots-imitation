"""Simple sensors related to the environment."""

import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import numpy as np
import typing

from envs.sensors import sensor

_ARRAY = typing.Iterable[float] 
_FLOAT_OR_ARRAY = typing.Union[float, _ARRAY] 
_DATATYPE_LIST = typing.Iterable[typing.Any]

class LastActionSensor(sensor.BoxSpaceSensor):
    """A sensor that reports the last action taken.

        Same idea as the IMUSensor class within the robot_sensors.py. 
        The LastActionSensor implements the BoxSpaceSensor and the Sensor prototype classes   
    """

    def __init__(self, 
                 num_actions: int, 
                 lower_bound: _FLOAT_OR_ARRAY = -1.0, 
                 upper_bound: _FLOAT_OR_ARRAY = 1.0, 
                 name: typing.Text = "LastAction", 
                 dtype: typing.Type[typing.Any] = np.float64) -> None:
        """LastActionSensor Constructor
        
        Args:
            num_actions: the number of actions to read
            lower_bound: the lower bound of the actions
            upper_bound: the upper bound of the actions
            name: the name of the sensor
            dtype: data type of sensor value        
        """
        self._num_actions = num_actions
        self._env = None

        # Initialize the superclass BoxSpaceSensor
        super(LastActionSensor, self).__init__(name=name, 
                                               shape=(self._num_actions,),
                                               lower_bound=lower_bound,
                                               upper_bound=upper_bound, 
                                               dtype=dtype)
    
    def on_reset(self, env):
        """From the callback, the sensor remembers the environment

        Args:
            env: the environment who invokes this callback       
        """
        self._env = env

    def _get_observation(self) -> _ARRAY:
        """Returns the last action of the environment"""
        # The environment will store the last action done by the robot
        return self._env.last_action        