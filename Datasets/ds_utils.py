import numpy as np
from random import random, randrange
# scale value from current range to target range
def scale(cMin, cMax, tMin, tMax, value): 
    value = (((value - cMin) * (tMax - tMin)) / (cMax - cMin)) + tMin
    return value

