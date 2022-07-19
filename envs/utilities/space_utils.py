"""Converts a list of sensors to gym space."""

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import gym
from gym import spaces
import numpy as np
import typing

from envs.sensors import sensor

class UnsupportedConversionError(Exception):
  """An exception when the function cannot convert sensors to the gym space."""


def convert_sensors_to_gym_space_dictionary(
    sensors: typing.List[sensor.Sensor]) -> gym.Space:
  """Convert a list of sensors to the corresponding gym space dictionary.

  Args:
    sensors: a list of the current sensors

  Returns:
    space: the converted gym space dictionary

  Raises:
    UnsupportedConversionError: raises when the function cannot convert the
      given list of sensors.
  """
  gym_space_dict = {}
  for s in sensors:
    if isinstance(s, sensor.BoxSpaceSensor):
      gym_space_dict[s.get_name()] = spaces.Box(np.array(s.get_lower_bound()),
                                                np.array(s.get_upper_bound()),
                                                dtype=np.float32)
    else:
      raise UnsupportedConversionError('sensors = ' + str(sensors))
  return spaces.Dict(gym_space_dict)
