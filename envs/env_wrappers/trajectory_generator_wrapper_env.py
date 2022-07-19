"""A wrapped envirnment with a trajectory generator (such as a pose offset in the simple_openloop wrapper)."""


class TrajectoryGeneratorWrapperEnv(object):
  """A wrapped environment with a trajectory generator (pose offset)."""

  def __init__(self, gym_env, trajectory_generator):
    """Initialzes the wrapped env.

    Args:
      gym_env: An instance of LocomotionGymEnv.
      trajectory_generator: A trajectory_generator that can potentially modify
        the action and observation. Typticall generators includes the PMTG and
        openloop signals. Expected to have get_action and get_observation
        interfaces.

    Raises:
      ValueError if the controller does not implement get_action and
      get_observation.

    """
    self._gym_env = gym_env
    # Make sure the trajectory_generator a get_action and get_observation function
    if not hasattr(trajectory_generator, 'get_action') or not hasattr(
        trajectory_generator, 'get_observation'):
      raise ValueError(
          'The controller does not have the necessary interface(s) implemented.'
      )

    self._trajectory_generator = trajectory_generator

    # The trajectory generator can subsume the action/observation space.
    if hasattr(trajectory_generator, 'observation_space'):
      self.observation_space = self._trajectory_generator.observation_space

    if hasattr(trajectory_generator, 'action_space'):
      self.action_space = self._trajectory_generator.action_space

  def __getattr__(self, attr):
    return getattr(self._gym_env, attr)

  def _modify_observation(self, observation):
    '''Simple open loop does not make a change to the observations'''
    return self._trajectory_generator.get_observation(observation)

  def reset(self, initial_motor_angles=None, reset_duration=0.0):
    # Simple open loop does not implement reset
    if getattr(self._trajectory_generator, 'reset'):
      self._trajectory_generator.reset()
    # Call reset function of the gym environment
    observation = self._gym_env.reset(initial_motor_angles, reset_duration)
    return self._modify_observation(observation)

  def step(self, action):
    """Steps the wrapped environment.

    Args:
      action: Numpy array. The input action from an NN agent.

    Returns:
      The tuple containing the modified observation, the reward, the epsiode end
      indicator.

    Raises:
      ValueError if input action is None.

    """

    if action is None:
      raise ValueError('Action cannot be None')
    
    # Add the initial offset to the predicted actions
    new_action = self._trajectory_generator.get_action(
        self._gym_env.robot.GetTimeSinceReset(), action)

    # Call the step function of the gym environment
    original_observation, reward, done, _ = self._gym_env.step(new_action)

    return self._modify_observation(original_observation), reward, done, _