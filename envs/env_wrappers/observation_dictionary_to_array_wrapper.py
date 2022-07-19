"""An env wrapper that flattens the observation dictionary to an array"""
import gym
from envs.utilities import env_utils


class ObservationDictionaryToArrayWrapper(gym.Env):
    '''An env wrapper that flattens the observation dictionary to an array'''
    def __init__(self, gym_env, observation_excluded=()):
        '''Initializes the wrapper'''
        self.observation_excluded = observation_excluded
        self._gym_env = gym_env
        self.observation_space = self._flatten_observation_spaces(
            self._gym_env.observation_space)
        self.action_space = self._gym_env.action_space
    
    def __getattr__(self, attr):
        return getattr(self._gym_env, attr)
    
    def _flatten_observation_spaces(self, observation_spaces):
        '''Flatten an observation space to an array
        
        The observational space is a dictionary with the lower and upper bound for each sensor.
        Take the upper and lower bounds and flatten into an array. 

        Ex:
            HistoricSensorWrapper(IMU):Box([-6.2831855 -6.2831855 -6.2831855 -6.2831855 -6.2831855 -6.2831855
                -6.2831855 -6.2831855 -6.2831855 -6.2831855 -6.2831855 -6.2831855
                -6.2831855 -6.2831855 -6.2831855 -6.2831855 -6.2831855 -6.2831855], 
            
                [6.2831855 6.2831855 6.2831855 6.2831855 6.2831855 6.2831855 6.2831855
                6.2831855 6.2831855 6.2831855 6.2831855 6.2831855 6.2831855 6.2831855
                6.2831855 6.2831855 6.2831855 6.2831855], (18,), float32),
            HistoricSensorWrapper(LastAction):Box([-1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1.
                -1. -1. -1. -1. -1. -1.], 
                [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.], (24,), float32))
            
            flattened:
                Box([-6.2831855 -6.2831855 -6.2831855 -6.2831855 -6.2831855 -6.2831855
                -6.2831855 -6.2831855 -6.2831855 -6.2831855 -6.2831855 -6.2831855
                -6.2831855 -6.2831855 -6.2831855 -6.2831855 -6.2831855 -6.2831855
                -1.        -1.        -1.        -1.        -1.        -1.
                -1.        -1.        -1.        -1.        -1.        -1.
                -1.        -1.        -1.        -1.        -1.        -1.
                -1.        -1.        -1.        -1.        -1.        -1.       ], 
                [6.2831855 6.2831855 6.2831855 6.2831855 6.2831855 6.2831855 6.2831855
                6.2831855 6.2831855 6.2831855 6.2831855 6.2831855 6.2831855 6.2831855
                6.2831855 6.2831855 6.2831855 6.2831855 1.        1.        1.
                1.        1.        1.        1.        1.        1.        1.
                1.        1.        1.        1.        1.        1.        1.
                1.        1.        1.        1.        1.        1.        1.       ], (42,), float32)
        '''
        flat_observation_space = env_utils.flatten_observation_spaces(
            observation_spaces=observation_spaces, 
            observation_excluded=self.observation_excluded)
        return flat_observation_space

    def _flatten_observation(self, input_observation):
        """Flatten the dictionary to an array."""
        return env_utils.flatten_observations(
            observation_dict=input_observation,
            observation_excluded=self.observation_excluded)

    def reset(self, initial_motor_angles=None, reset_duration=0.0):
        observation = self._gym_env.reset(
            initial_motor_angles=initial_motor_angles,
            reset_duration=reset_duration)
        return self._flatten_observation(observation)

    def step(self, action):
        """Steps the wrapped environment.

        Args:
        action: Numpy array. The input action from an NN agent.

        Returns:
        The tuple containing the flattened observation, the reward, the epsiode
            end indicator.
        """
        observation_dict, reward, done, _ = self._gym_env.step(action)
        
        return self._flatten_observation(observation_dict), reward, done, _

    def render(self, mode='human'):
        return self._gym_env.render(mode)