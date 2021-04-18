from enum import Enum
from operator import index
import os
import sys
# to import files from sibling subfolders, you need to basically search from the top
# so this adds the shallowmind folder to the paths it searches for
# honestly, we should maybe restructure this as a package maybe so this is easier?
# ^ tried that, didnt work but i probably did it wrong or maybe thats just not how it works
path_to_file = os.path.abspath(os.path.dirname(__file__))
index_in_path = path_to_file.rfind("ShallowMind") + len("ShallowMind") # get path after "ShallowMind" (i cant count)
sm_path = path_to_file[:index_in_path]
sys.path.insert(0, sm_path)

from Utils import seeding
from Datasets import Ellipse, GaussianBoundary

"""Gets a dataset of a specific type for the provided options

:param dataType: ds.DataTypes(.name) - the datatype to get
:param options: ds.DataTypes(.options) - the options for the dataset. If not provided, defaults are used
:returns: np.ndarray - the dataset in [[[x], [y]], label] format
"""
def getDataset(dataType, options=None):
    _ = 0
    if dataType == DataTypes.ELLIPSE:
        options = DataTypes.EllipseOptions() if options is None else options
        dataset = Ellipse.getPoints(
            numPoints=options.numPoints,
            seed=options.seed,
            noise=options.noise,
            width=options.width,
            height=options.height,
            angle=options.angle,
            center=options.center,
            vMin=options.vMin,
            vMax=options.vMax)
    
    elif dataType is DataTypes.GAUSSIAN_BOUNDARY:
        options = DataTypes.GaussianBoundaryOptions() if options is None else options
        dataset = GaussianBoundary.getPoints(
            coVec=options.coVec,
            seed=options.seed,
            numPoints=options.numPoints,
            sigma=options.sigma,
            peak=options.peak,
            xMin=options.xMin,
            xMax=options.xMax,
            yMin=options.yMin,
            yMax=options.yMax)
    
    return dataset
"""
Noise generation:
    1. distance:
        If the function is on a graph with values from 0-10,
        how far from the function boundary should the noise occur
    2. chance
        What chance should the points within the noise distance have of
        performing a coin flip on which value to take
    
    Chance could have also been generated as:
        What chance should the points within the noise distance have of
        flipping their value
    But I changed this because that results in layers around the function if its
    too high.
"""
class DataTypes():
    ELLIPSE = "ellipse"
    GAUSSIAN_BOUNDARY = "gaussian_boundary"

    class EllipseOptions():
        name = "ellipse"
        # self explanatory, look in ellipse class for info on noise (distance/chance)
        # vMin, vMax is size of sample
        def __init__(
          self, numPoints=2000, seed=seeding.getSeed(), distance=2, chance=0.5, center=(0,0),
          width=10, height=13, angle=0, vMin=-10, vMax=10):
            self.numPoints = numPoints
            self.seed = seed
            self.distance = distance
            self.chance = chance
            self.noise = self.distance, self.chance
            self.center = center
            self.width = width
            self.height = height
            self.angle = angle
            self.vMin = vMin
            self.vMax = vMax
    
    class GaussianBoundaryOptions():
        name = "gaussian_boundary"
        def __init__(self,
          coVec=[.25, 0, -5], seed=seeding.getSeed(), numPoints=2000, sigma=.2,
          peak=.075, xMin=-10, xMax=10, yMin=-10, yMax=10):
            self.coVec = coVec
            self.seed=seed
            self.numPoints = numPoints
            self.sigma = sigma
            self.peak =  peak
            self.xMin = xMin
            self.xMax = xMax
            self.yMin = yMin
            self.yMax = yMax


# options = DataTypes.GaussianBoundaryOptions()
# name = DataTypes.GAUSSIAN_BOUNDARY
# points = getDataset(name, options)
