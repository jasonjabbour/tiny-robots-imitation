class TinyRobot:
    def __init__(self, robot):
        self._robot = robot

    def get_robot(self):
        return self._robot
    
    def set_robot(self, robot):
        self._robot = robot
