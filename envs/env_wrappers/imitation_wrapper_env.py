"""A wrapper for motion imitation environment.
    Specifically for extending the observational space
    with future motion reference frames. 
"""

import gym
import numpy as np

class ImitationWrapperEnv(object):
  """An env using for training policy with motion imitation."""

  def __init__(self,
      gym_env,
      episode_length_start=1000,
      episode_length_end=1000,
      curriculum_steps=0,
      num_parallel_envs=1,
      include_future_frames=False):
    """Initialzes the wrapped env.

    Args:
      gym_env: An instance of LocomotionGymEnv.
      include_future_frames: If false don't add future frames to observational space
    """
    self._gym_env = gym_env
    self.include_future_frames = include_future_frames
    # Extend observational space to include future frames
    self.observation_space = self._build_observation_space()

    self._episode_length_start = episode_length_start
    self._episode_length_end = episode_length_end
    self._curriculum_steps = int(np.ceil(curriculum_steps / num_parallel_envs))
    self._total_step_count = 0

    if self._enable_curriculum():
      self._update_time_limit()

    self.seed()
    return

  def __getattr__(self, attr):
    return getattr(self._gym_env, attr)

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
    original_observation, reward, done, original_info = self._gym_env.step(action)
    observation = self._modify_observation(original_observation)
    terminated = done

    done |= (self.env_step_counter >= self._max_episode_steps)

    if not done:
      self._total_step_count += 1

    info = {"terminated": terminated, "processed_action": original_info['processed_action']}

    return observation, reward, done, info

  def reset(self, initial_motor_angles=None, reset_duration=0.0):
    """Resets the robot's position in the world or rebuild the sim world.

    The simulation world will be rebuilt if self._hard_reset is True.

    Args:
      initial_motor_angles: A list of Floats. The desired joint angles after
        reset. If None, the robot will use its built-in value.
      reset_duration: Float. The time (in seconds) needed to rotate all motors
        to the desired initial values.

    Returns:
      A numpy array contains the initial observation after reset.
    """
    original_observation = self._gym_env.reset(initial_motor_angles, reset_duration)
    observation = self._modify_observation(original_observation)

    if self._enable_curriculum():
      self._update_time_limit()

    return observation
  
  def _modify_observation(self, original_observation):
    """Appends target observations from the reference motion to the observations.

    Args:
      original_observation: A numpy array containing the original observations.

    Returns:
      A numpy array contains the initial original concatenated with target
      observations from the reference motion.
    """
    # Don't include 4 future frames
    if not self.include_future_frames:
      # Only return observational space created by sensors
      return original_observation

    target_observation = self._task.build_target_obs()
    observation = np.concatenate([original_observation, target_observation], axis=-1)
    return observation

  def _build_observation_space(self):
    """Constructs the observation space, including target observations from
    the reference motion.

    Returns:
      Observation space representing the concatenations of the original
      observations and target observations.
    """
    obs_space0 = self._gym_env.observation_space
    low0 = obs_space0.low
    high0 = obs_space0.high

    # Don't extend observational space with 4 future frames
    if not self.include_future_frames:
      #instead of returning an observational space extended by 60 dimensions, return original 60 dimensions of only sensors
      return obs_space0

    # Get the target observation bounds from task to extend observational space
    task_low, task_high = self._task.get_target_obs_bounds()
    low = np.concatenate([low0, task_low], axis=-1)
    high = np.concatenate([high0, task_high], axis=-1)

    obs_space = gym.spaces.Box(low, high)

    return obs_space

  def _enable_curriculum(self):
    """Check if curriculum is enabled."""
    return self._curriculum_steps > 0

  def _update_time_limit(self):
    """Updates the current episode length depending on the number of environment steps taken so far."""
    t = float(self._total_step_count) / self._curriculum_steps
    t = np.clip(t, 0.0, 1.0)
    t = np.power(t, 3.0)
    new_steps = int((1.0 - t) * self._episode_length_start +
                       t * self._episode_length_end)
    self._max_episode_steps = new_steps
    return
