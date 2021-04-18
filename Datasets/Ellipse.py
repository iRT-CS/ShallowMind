from random import random, randrange
from numpy.lib.arraysetops import intersect1d
from numpy.random import seed as np_seed
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib import cm

"""Generates an ellipse dataset
    
    Noise generation:
        1. distance:
            If the ellipse is on a graph with values from 0-10,
            how far from the ellipse boundary should the noise occur
        2. chance
            What chance should the points within the noise distance have of
            performing a coin flip on which value to take
        
        Chance could have also been generated as:
            What chance should the points within the noise distance have of
            flipping their value
        But I changed this because that results in circular layers around the ellipse if its
        too high.
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
def getPoints(numPoints:int, seed:int, noise:tuple, width:int, height:int,
angle:int=0, center:tuple=(0,0), vMin:int= -10, vMax:int=10) -> np.ndarray:

    distance, chance = noise
    # scale down to 0,1
    distance = scale(0, 10, 0, 1, distance)

    # generate random points
    # use to be normal distribution but it was wonky
    data_rng = np.random.default_rng(seed)
    points = data_rng.random(size=(2, numPoints))
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
        points[0][index] = scale(xMin, xMax, vMin, vMax, points[0][index])
        points[1][index] = scale(yMin, yMax, vMin, vMax, points[1][index])
    
    # get distance from ellipse to point
    # boundary line is 1
    # results in array with numpoints points with each value as distance to ellipse
    cos_angle = np.cos(np.radians(180.-angle))
    sin_angle = np.sin(np.radians(180.-angle))
    xc = points[0] - center[0]
    yc = points[1] - center[1]
    xct = xc * cos_angle - yc * sin_angle
    yct = xc * sin_angle + yc * cos_angle
    rad_cc = (xct**2/(width/2.)**2) + (yct**2/(height/2.)**2)

    # fill outside with 1, inside with 0
    labels = np.full(numPoints, 1)
    labels[np.where(rad_cc <= 1)[0]] = 0
    
    # apply noise

    # get values within noise range
    noiseRange = intersect1d(np.where(rad_cc > 1-distance), np.where(rad_cc < 1 + distance))

    # these were used for flip
    # underNoise = intersect1d(np.where(rad_cc <= 1), np.where(rad_cc > 1-distance))
    # overNoise = intersect1d(np.where(rad_cc > 1), np.where(rad_cc < 1+distance))
    # probLambda = lambda x: abs(int(chance + random()) - labels[x]) # chance to flip value, not used

    # chance to 50/50
    def chanceForNoise(current):
        if (chance + random() >= 1):
            return abs(int(0.5 + random()))
        return labels[current]
    probVectorized = np.vectorize(chanceForNoise)
    if len(noiseRange) > 0:
        labels[noiseRange] = probVectorized(noiseRange)
    # labels[underNoise] = probVectorized(underNoise)
    # labels[overNoise] = probVectorized(overNoise)
    dataset = np.array([points, []], dtype=object)
    dataset[1] = labels
    return dataset

# plots an ellpise from a dataset with a background ellipse for comparison
# for a description of params, see getPoints()
def plotEllipse(numPoints, seed, noise, width, height, angle=0, center=(0,0), vMin= -10, vMax=10, dataset=None):
    cmap = cm.get_cmap("coolwarm")
    # data_rng = np.random.default_rng(seed)
    # points = data_rng.normal(size=(2, numPoints))
    fig = plt.figure()
    ax = fig.add_subplot()
    ellipse = patches.Ellipse(center, width, height, angle, zorder=0, facecolor=cmap(0.2))
    ax.add_patch(ellipse)
    # labels = np.array(["#FF0000"] * len(points[0]))
    # for index, point in enumerate(points):
    #     if ellipse.contains_point(point):
    #         labels[index] = "#0000FF"
    points = dataset[0]
    labels=dataset[1]
    plt.scatter(points[0], points[1], c=labels, zorder=1, cmap=cmap)
    plt.show()

# scale value from current range to target range
def scale(cMin, cMax, tMin, tMax, value): 
    value = (((value - cMin) * (tMax - tMin)) / (cMax - cMin)) + tMin
    return value

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
#     numPoints=numPoints,
#     seed=seed,
#     noise=noise,
#     center=center,
#     width=width,
#     height=height,
#     angle=angle,
#     vMin=vMin,
#     vMax=vMax,
#     dataset=dataset
# )