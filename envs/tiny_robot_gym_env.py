
import gym
from gym.utils import seeding

class TinyRobotGymEnv(gym.Env):
    """Open AI Gym Environment for Locomotion Tasks"""

    #Store additional information about gym environment class
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 100
    }

    def __init__(self, robot=None):
        """Initialize Open AI Gym Environment"""

        self.seed()
        self._robot = robot
    

    def seed(self, seed=None):
        self.np_random, self.np_random_seed = seeding.np_random(seed)
        return [self.np_random_seed]

    def reset(self):
        pass

    def step(self):
        pass

    def render(self):
        pass

