import math 
import copy

TWO_PI = 2 * math.pi

def MapToMinusPiToPi(angles):
    """Maps a list of angles to [-pi, pi].
        If more than 3.14 or -3.14 then use negative counter part. 
        Look at Unit circle :)

    Args:
        angles: A list of angles in rad.

    Example:
        Input: [3.14, 3.5, 6.28, 9.56, -1.57]
        Ouput: [3.14, -2.7831853071795862, -0.0031853071795859833, -3.006370614359172]

    Returns:
        A list of angle mapped to [-pi, pi].
    """
    #Create a copy of the angle list
    mapped_angles = copy.deepcopy(angles)

    for i in range(len(angles)):
        mapped_angles[i] = math.fmod(angles[i], TWO_PI)
        if mapped_angles[i] >= math.pi:
            mapped_angles[i] -= TWO_PI
        elif mapped_angles[i] < -math.pi:
            mapped_angles[i] += TWO_PI
    return mapped_angles

if __name__ == '__main__':
    print(MapToMinusPiToPi([3.14, 3.5, 6.28, 9.56, -1.57]))