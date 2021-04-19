from random import random, randrange
from matplotlib.transforms import Bbox
from numpy.lib.arraysetops import intersect1d
from numpy.random import seed as np_seed
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib import cm
import matplotlib as mpl
import os, sys
path_to_file = os.path.abspath(os.path.dirname(__file__))
index_in_path = path_to_file.rfind("ShallowMind") + len("ShallowMind") # get path after "ShallowMind" (i cant count)
sm_path = path_to_file[:index_in_path]
sys.path.insert(0, sm_path)
from Datasets import ds_utils

"""Generates an ellipse dataset

:param numPoints: int - the number of points to generate
:param seed: int - the seed of the dataset for reproducabiity (noise doesnt care about seed tho, someone can fix that)
:param noise: tuple (distance, chance) - see above
:param width: int - width of the ellipse
:param height: int - the height of the ellipse
:param angle: int - angle of the ellipse
:param center: tuple (x, y) - the center point of the tuple
:param vMin: int - minimum range of dataset
:param vMax: int - max range of dataset
:returns: np.ndarray: [[x_array, y_array], [label_array]] the dataset with labels
"""
def getPoints(numPoints:int, seed:int, noise:tuple, vMin:int, vMax:int, width:int, height:int,
angle:int, center:tuple) -> np.ndarray:

    distance, chance = noise
    # scale down to 0,1
    distance = ds_utils.scale(0, 10, 0, 1, distance)

    # generate random points
    # use to be normal distribution but it was wonky
    data_rng = np.random.default_rng(seed)
    points = data_rng.random(size=(numPoints, 2))
    x_points = points[:,0]
    y_points = points[:,1]
    # here for testing with other distributions since they arent 0,1
    # xMin = min(points[0])
    # xMax = max(points[0])
    # yMin = min(points[1])
    # yMax = max(points[1])
    xMin = 0
    xMax = 1
    yMin = 0
    yMax = 1
    # scale to vMin, vMax
    for index in range(numPoints):
        x_points[index] = ds_utils.scale(xMin, xMax, vMin, vMax, x_points[index])
        y_points[index] = ds_utils.scale(yMin, yMax, vMin, vMax, y_points[index])
    
    # get distance from ellipse to point
    # boundary line is 1
    # results in array with numpoints points with each value as distance to ellipse
    cos_angle = np.cos(np.radians(180.-angle))
    sin_angle = np.sin(np.radians(180.-angle))
    xc = x_points - center[0]
    yc = y_points - center[1]
    xct = xc * cos_angle - yc * sin_angle
    yct = xc * sin_angle + yc * cos_angle
    rad_cc = (xct**2/(width/2.)**2) + (yct**2/(height/2.)**2)

    # fill outside with 0, inside with 1
    # RED 0 BLUE 1
    labels = np.full(numPoints, 0)
    labels[np.where(rad_cc <= 1)[0]] = 1
    
    # apply noise

    # get values within noise range
    noiseRange = intersect1d(np.where(rad_cc > 1-distance), np.where(rad_cc < 1 + distance))

    # these were used for flip
    # underNoise = intersect1d(np.where(rad_cc <= 1), np.where(rad_cc > 1-distance))
    # overNoise = intersect1d(np.where(rad_cc > 1), np.where(rad_cc < 1+distance))
    # probLambda = lambda x: abs(int(chance + random()) - labels[x]) # chance to flip value, not used

    # chance to 50/50
    def chanceForNoiseUnvectorized(x):
        if (chance + random() >= 1):
            return abs(int(0.5 + random()))
        return labels[x]
    chanceForNoise = np.vectorize(chanceForNoiseUnvectorized)

    if len(noiseRange) > 0:
        labels[noiseRange] = chanceForNoise(noiseRange) #broke
    # labels[underNoise] = probVectorized(underNoise)
    # labels[overNoise] = probVectorized(overNoise)
    dataset = np.array([points, []], dtype=object)
    dataset[1] = labels
    return dataset

# plots an ellpise from a dataset with a background ellipse for comparison
# for a description of params, see getPoints()
def plotEllipse(width, height, angle=0, center=(0,0), vMin= -10, vMax=10, dataset=None):
    cmap = cm.get_cmap("RdYlBu")
    # data_rng = np.random.default_rng(seed)
    # points = data_rng.normal(size=(2, numPoints))
    fig = plt.figure()
    ax = fig.add_subplot()
    ellipse = patches.Ellipse(center, width, height, angle, zorder=0, facecolor=cmap(.8))
    ax.add_patch(ellipse)
    # labels = np.array(["#FF0000"] * len(points[0]))
    # for index, point in enumerate(points):
    #     if ellipse.contains_point(point):
    #         labels[index] = "#0000FF"
    points = dataset[0]
    labels=dataset[1]
    plt.scatter(points[:,0], points[:,1], c=labels, zorder=1, cmap=cmap)
    plt.show()

"""Gets the boundary of an ellipse as a matplotlib patch
see getPoints for an explanation of ellipse params
:param linewidth: float - the thickness of the line
:param linecolor: string - the color of the line
"""
def getBoundary(vMin, vMax, width, height, angle, center, linewidth, linecolor):
    patch = patches.Ellipse(
        xy=center,
        width=width, 
        height=height,
        angle=angle,
        zorder=4,
        fill=False,
        linewidth=linewidth,
        color=linecolor,)
    return patch

"""Sets the boundary onto an axis
see getPoints for an explanation of ellipse params
:param linewidth: float - the thickness of the line
:param linecolor: string - the color of the line
"""
def setBoundary(ax, vMin, vMax, width, height, angle, center, linewidth, linecolor):
    patch = getBoundary(
        width=width,
        height=height,
        angle=angle,
        center=center,
        vMin=vMin,
        vMax=vMax,
        linewidth=linewidth,
        linecolor=linecolor)
    ax.add_patch(patch)


# numPoints=2000
# seed=2
# distance = 2
# chance = 0.5
# noise = distance, chance
# center = 0, 0
# width = 10
# height = 20
# angle = 45
# vMin = -10
# vMax = 10

# dataset = getPoints(
#     numPoints=numPoints,
#     seed=seed,
#     noise=noise,
#     center=center,
#     width=width,
#     height=height,
#     angle=angle,
#     vMin=vMin,
#     vMax=vMax
# )
# plotEllipse(
#     center=center,
#     width=width,
#     height=height,
#     angle=angle,
#     vMin=vMin,
#     vMax=vMax,
#     dataset=dataset
# )