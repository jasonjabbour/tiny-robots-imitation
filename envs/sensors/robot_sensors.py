import os 
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import numpy as np
import typing

from envs.sensors import sensor

_ARRAY = typing.Iterable[float] #pylint: disable=invalid-name
_FLOAT_OR_ARRAY = typing.Union[float, _ARRAY] #pylint: disable=invalid-name
_DATATYPE_LIST = typing.Iterable[typing.Any] #pylint: disable=invalid-name

class IMUSensor(sensor.BoxSpaceSensor):
    """An IMU sensor that reads orientations and angular velocities"""

    def __init__(self, 
                channels: typing.Iterable[typing.Text] = None, 
                noisy_reading: bool = True, 
                lower_bound: _FLOAT_OR_ARRAY = None, 
                upper_bound: _FLOAT_OR_ARRAY = None, 
                name: typing.Text = "IMU", 
                dtype: typing.Type[typing.Any] = np.float64) -> None:
        """IMU Sensor Constructor
        
        Generates separate IMU value channels, e.g. IMU_R, IMU_P, IMU_dR, ...

        Args:
            channels: value channels wants to subscribe A upper letter represents
                orientation and a lower letter represents angular velcoity. (e.g. ['R', 
                'P', 'Y', 'dR', 'dP', 'dY'] or ['R', 'P', 'dR', 'dP'])
                noisy_reading: whether values are true observations
                lower_bound: the lower bound IMU values 
                    (default: [-2pi, -2pi, -2000pi, -2000pi])
                upper_bound: the lower bound IMU values
                    (default: [2pi, 2pi, 2000pi, 2000pi])
                name: the name of the sensor
                dtype: data type of sensor value        
        """
        self._channels = channels if channels else ['R', 'P', 'Y']
        self._num_channels = len(self._channels)
        self._noisy_reading = noisy_reading

        # Compute the default lower and upper bounds
        if lower_bound is None and upper_bound is None:
            lower_bound = []
            upper_bound = []
            for channel in self._channels:
                if channel in ['R', 'P', 'Y']:
                    lower_bound.append(-2.0 * np.pi)
                    upper_bound.append(2.0 * np.pi)
                elif channel in ['Rcos', 'Rsin', 'Pcos', 'Psin', 'Ycos', 'Ysin']:
                    lower_bound.append(-1.)
                    upper_bound.append(1.)
                elif channel in ['dR', 'dP', 'dY']:
                    lower_bound.append(-2000.0 * np.pi)
                    upper_bound.append(2000.0 * np.pi)
        
        # BoxSpaceSensor Prototype class which is a subclass to the Sensor class 
        super(IMUSensor, self).__init__(
            name=name,
            shape=(self._num_channels, ), 
            lower_bound=lower_bound, 
            upper_bound=upper_bound,
            dtype=dtype)
        
        # Compute the observation_datatype
        datatype = [('{}_{}'.format(name, channel), self._dtype) 
                    for channel in self._channels]

        self._datatype = datatype

    def get_channels(self) -> typing.Iterable[typing.Text]:
        return self._channels

    def get_num_channels(self) -> int:
        return self._num_channels

    def get_observation_datatype(self) -> _DATATYPE_LIST:
        """Returns box-shape data type."""
        return self._datatype

    def _get_observation(self) -> _ARRAY:
        if self._noisy_reading:
            rpy = self._robot.GetBaseRollPitchYaw()
            #drpy = self._robot.GetBaseRollPitchYawRate()
        else:
            rpy = self._robot.getTrueBaseRollPitchYaw()
            #drpy = self._robot.getTrueBaseRollPitchYawRate()
        
        assert len(rpy) >=3, rpy
        #assert len(drpy) >=3, drpy
    
        observations = np.zeros(self._num_channels)
        for i, channel in enumerate(self._channels):
            if channel == "R":
                observations[i] = rpy[0]
            if channel == "Rcos":
                observations[i] = np.cos(rpy[0])
            if channel == "Rsin":
                observations[i] = np.sin(rpy[0])
            if channel == "P":
                observations[i] = rpy[1]
            if channel == "Pcos":
                observations[i] = np.cos(rpy[1])
            if channel == "Psin":
                observations[i] = np.sin(rpy[1])
            if channel == "Y":
                observations[i] = rpy[2]
            if channel == "Ycos":
                observations[i] = np.cos(rpy[2])
            if channel == "Ysin":
                observations[i] = np.sin(rpy[2])
            # if channel == "dR":
            #     observations[i] = drpy[0]
            # if channel == "dP":
            #     observations[i] = drpy[1]
            # if channel == "dY":
            #     observations[i] = drpy[2]
        
        return observations