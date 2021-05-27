from enum import Enum
from operator import index
import os
from re import S
import sys
import numpy as np
# to import files from sibling subfolders, you need to basically search from the top
# so this adds the shallowmind folder to the paths it searches for
# honestly, we should maybe restructure this as a package maybe so this is easier?
# ^ tried that, didnt work but i probably did it wrong or maybe thats just not how it works
path_to_file = os.path.abspath(os.path.dirname(__file__))
index_in_path = path_to_file.rfind("ShallowMind") + len("ShallowMind") # get path after "ShallowMind" (i cant count)
sm_path = path_to_file[:index_in_path]
sys.path.insert(0, sm_path)

from Utils import seeding
from Datasets import Ellipse, GaussianBoundary, Polynomial


def getDataset(dataType=None, options=None) -> np.ndarray:
    """Gets a dataset of a specific type for the provided options

    Args:
        dataType: ds.DataTypes(.name) - the datatype to get
        options: ds.DataTypes(.options) - the options for the dataset. If not provided, defaults are used
        
        returns: np.ndarray - the dataset in [[[x], [y]], label] format
    """
    dataType = options.name if dataType is None else dataType
    if dataType == EllipseOptions.name:
        options = EllipseOptions() if options is None else options
        dataset = Ellipse.getPoints(
            numPoints=options.numPoints,
            seed=options.seed,
            noise=options.noise,
            vMin=options.vMin,
            vMax=options.vMax,
            width=options.width,
            height=options.height,
            angle=options.angle,
            center=options.center)

    elif dataType == PolynomialOptions.name:
        options = PolynomialOptions() if options is None else options
        dataset = Polynomial.getPoints(
            numPoints=options.numPoints,
            seed=options.seed,
            noise=options.noise,
            vMin=options.vMin,
            vMax=options.vMax,
            coefficients=options.coefficients)
    
    # deprecated, use POLYNOMIAL
    elif dataType == DataTypes.GAUSSIAN_BOUNDARY:
        options = GaussianBoundaryOptions() if options is None else options
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
    else:
        raise ValueError("Either a data type or data options must be provided and valid")
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
    POLYNOMIAL = "polynomial"
    GAUSSIAN_BOUNDARY = "gaussian_boundary"

    def setNoise(self, distance, chance):
        """Sets noise for a dataset
        All other variables can be set directly, but noise components cant
        unless you chance the actual noise variable, so this exists to make it easier

        Args:
            distance:int - see noise generation comment
            chance:int - see noise generation comment
        """
        self.noise = (distance, chance)
    
    def getInfoDict(self) -> dict:
        """Gets a dictionary of the attributes for the dataset options

        Returns:
            dict - the dataset options in a dictionary with their attribute and value
        """
        attributes = dir(self)
        attr_dict = {}
        attr_dict["__class__"] = getattr(self, "__class__")
        # doesnt include these
        dont_include = ["getInfoDict", "ELLIPSE", "POLYNOMIAL", "GAUSSIAN_BOUNDARY", "setNoise"]

        for attr in attributes:
            if attr[0:2] != "__" and attr not in dont_include:
                attr_dict[attr] = getattr(self, attr)
        return attr_dict



class EllipseOptions(DataTypes):
    name = DataTypes.ELLIPSE
    # self explanatory, look in ellipse class for info on noise (distance/chance)
    # vMin, vMax is size of sample
    def __init__(
        self, numPoints=2000, seed=seeding.getSeed(), distance=2, chance=0.5, vMin=-10, vMax=10,
        center=(0,0), width=10, height=13, angle=0):
        super().__init__()
        self.numPoints = numPoints
        self.seed = seed
        self.distance = distance
        self.chance = chance
        self.noise = self.distance, self.chance
        self.vMin = vMin
        self.vMax = vMax
        self.center = center
        self.width = width
        self.height = height
        self.angle = angle


    

# look in Polynomial class for info
class PolynomialOptions(DataTypes):
    name = DataTypes.POLYNOMIAL
    def __init__(self, numPoints=2000, seed=seeding.getSeed(), distance=1, chance=0.5, vMin=-10, vMax=10,
        coefficients=[.25, 0, -2]):
        super().__init__()
        self.numPoints = numPoints
        self.seed = seed
        self.distance = distance
        self.chance = chance
        self.noise = self.distance, self.chance
        self.vMin = vMin
        self.vMax = vMax
        self.coefficients = coefficients

# use Polynomial
class GaussianBoundaryOptions(DataTypes):
    name = DataTypes.GAUSSIAN_BOUNDARY
    def __init__(self,
      coVec=[.25, 0, -5], seed=seeding.getSeed(), numPoints=2000, sigma=.2,
      peak=.075, xMin=-10, xMax=10, yMin=-10, yMax=10):
        super().__init__()
        self.coVec = coVec
        self.seed=seed
        self.numPoints = numPoints
        self.sigma = sigma
        self.peak =  peak
        self.xMin = xMin
        self.xMax = xMax
        self.yMin = yMin
        self.yMax = yMax


# options = PolynomialOptions()
# name = DataTypes.POLYNOMIAL
# points = getDataset(name, options)

def setDatasetBoundaryPlot(ax, dataType=None, options=None, linewidth=1, linecolor="#1967b0"):
    dataType = options.name if dataType is None else dataType
    if dataType is DataTypes.ELLIPSE:
        options = EllipseOptions() if options is None else options
        Ellipse.setBoundary(
            ax=ax,
            vMin=options.vMin,
            vMax=options.vMax,
            width=options.width,
            height=options.height,
            angle=options.angle,
            center=options.center,
            linewidth=linewidth,
            linecolor=linecolor)
    
    elif dataType is DataTypes.POLYNOMIAL:
        options = PolynomialOptions() if options is None else options
        Polynomial.setBoundary(
            ax=ax,
            vMin=options.vMin,
            vMax=options.vMax,
            coefficients=options.coefficients,
            linewidth=linewidth,
            linecolor=linecolor)
    else:
        raise ValueError("Either a data type or data options must be provided and valid")

    ax.set_ylim(options.vMin-0.5, options.vMax+0.5)
    ax.set_xlim(options.vMin-0.5, options.vMax+0.5)

# dataset = EllipseOptions()
# print(dataset.getInfoDict())