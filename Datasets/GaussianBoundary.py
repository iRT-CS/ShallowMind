import numpy as np
from scipy.optimize import fmin_cobyla
import math
import random
from Utils.distanceToCurve import distanceToCurve
from Utils import seeding
#import timeit

# DEPRECATED, USE Polynomial.py

# seed for reproduceable datasets
seedNum = seeding.getSeed()

#Vector function
def genFunctionUniform(degree = 2,minimum = -7,maximum = 7):
    coefficients = []
    for i in range(degree+1):
        random.seed(seedNum)
        coefficient = random.randrange(minimum,maximum)
        coefficients.insert(0,coefficient)
    return coefficients

def genFunctionGaussian(degree = 2,mean = 0,sigma = 7, scaling = False):
    coefficients = []
    for i in range(degree+1):
        if(scaling):
            eSigma = sigma/(i+1)
        else:
            eSigma = sigma
        coefficient = gauss(mean,eSigma)
        coefficients.append(coefficient)
    return coefficients

def evalFunction(coVec,ipVar):
    boundaryPoint = 0
    maximum = len(coVec) - 1
    for i in range(len(coVec)):
        power = maximum - i
        termVal = pow(ipVar,power)*coVec[power]
        boundaryPoint += termVal
    return boundaryPoint
def pointDistance(x1,y1,x2,y2):
    return np.sqrt(pow((x1-x2),2)+pow((y1-y2),2))
'''
def distanceToCurve(coVec,ipVar,dpVar):
    maxDistance = abs(dpVar - evalFunction(coVec, ipVar))
    xMin = fmin_cobyla(\
        lambda x: pointDistance(ipVar,dpVar,x[0],evalFunction(coVec, x[0]))\
        ,x0 = [ipVar,ipVar],\
        cons = [lambda x: abs(ipVar-maxDistance)],\
        rhoend = 1e-3)
    return pointDistance(ipVar,dpVar,xMin[0], evalFunction(coVec, xMin[0]))
'''
def gauss(distance, sigma):
    sigmaComponent = pow(sigma,2)*2
    #denominator = math.sqrt(math.pi*sigmaComponent)
    #I want to control the max height
    denominator = 1
    if( sigmaComponent == 0):
        return 0
    numerator = math.exp(-pow(distance,2)/sigmaComponent)
    return numerator/denominator

# [1,1] y = x + 0, y = x + 1
def getPoints(coVec,numPoints,sigma,peak,xMin,xMax,yMin,yMax, seed=seeding.getSeed()):
    random.seed(seed)
    x = np.random.rand(numPoints)
    xRange = xMax - xMin
    x = list(map(lambda v: (v*xRange)+xMin,x))
    random.seed(seed*2)
    y = np.random.rand(numPoints)
    yRange = yMax - yMin
    y = list(map(lambda v: (v*yRange)+yMin,y))
    boundaryPoints = np.polyval(coVec,x)
    distances = list(map(lambda m,n: distanceToCurve(coVec,m,n),x,y))
    distances = list(map(lambda x:peak*x,distances))
    gaussian = list(map(lambda d: gauss(d,sigma),distances))
    random.seed(seed)
    flip = list(map(lambda g: (np.random.uniform()<g),gaussian))
    cleanVals = list(map(lambda d,b: (d>b),y,boundaryPoints))
    dirtyVals = list(map(lambda v,f: v^f, cleanVals,flip))
    points = list(map(lambda i,d,v: [i,d,v],x,y,dirtyVals))
    
    npPoints = np.array(points)
    labels = npPoints[:,2]
    pairs = np.transpose([x,y])
    points = (pairs, labels)
    return points
