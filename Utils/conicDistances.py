import numpy as np
from scipy.optimize import fmin_cobyla
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fmin_cobyla
import math
import random
from Utils.utils import iterate,plotData

def pol2car(r,theta):
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    return x,y

def car2pol(x,y):
    r = np.sqrt((x**2)+(y**2))
    theta = math.atan2(y,x)
    return r,theta

def evalEllipse(coVec,theta):
    num = coVec[0]*coVec[1]
    den = np.sqrt(((coVec[1]*np.cos(theta))**2)+((coVec[0]*np.sin(theta))**2))
    return num/den

def lineConstraint(theta,c,d,coVec):
    a = coVec[0]
    b = coVec[1]
    r = evalEllipse(coVec,theta)
    x,y = pol2car(r,theta)
    ySide = (b**2)*x*(y-d)
    xSide = (a**2)*y*(x-c)
    return ySide-xSide

# other constraint: lambda x:0.0001-abs(lineConstraint(x[0],x1,y1,coVec))
def getEDistance(coVec,x1,y1):
    r,theta = car2pol(x1,y1)
    eTheta = fmin_cobyla(lambda x: abs(r-evalEllipse(coVec,x[0])),x0 = [theta,theta],cons = [lambda x: 6+x[0],lambda x: 14-x[0]], rhoend = 1e-4)[1]
    # Perpindicular line constraint
    eR = evalEllipse(coVec,eTheta)
    eX, eY = pol2car(eR,eTheta)
    dX = x1-eX
    dY = y1-eY
    dist = np.sqrt((dX**2)+(dY**2))
    return dist

# print(getEDistance([1,1],2,0))

coVec = [5,10]

def gauss(distance, sigma):
    sigmaComponent = pow(sigma,2)*2
    #denominator = math.sqrt(math.pi*sigmaComponent)
    #I want to control the max height
    denominator = 1
    if( sigmaComponent == 0):
        return 0
    numerator = math.exp(-pow(distance,2)/sigmaComponent)
    return numerator/denominator

def getPointsConic(coVec,numPoints,sigma,peak,xMin,xMax,yMin,yMax):
    x = np.random.rand(numPoints)
    xRange = xMax - xMin
    x = list(map(lambda v: (v*xRange)+xMin,x))
    y = np.random.rand(numPoints)
    yRange = yMax - yMin
    y = list(map(lambda v: (v*yRange)+yMin,y))
    print(x)
    print(y)
    pairs = np.array(list(map(lambda x,y: list(car2pol(x,y)),x,y)))
    print(pairs)
    print(pairs.shape)
    r = pairs[:,0]
    theta = pairs[:,1]
    print(r)
    print(theta)
    # write boundarypts in terms of r & theta
    boundaryR = list(map(lambda t:evalEllipse(coVec,t),theta))
    distances = list(map(lambda m,n: getEDistance(coVec,m,n),x,y))
    distances = list(map(lambda x:peak*x,distances))
    gaussian = list(map(lambda d: gauss(d,sigma),distances))
    flip = list(map(lambda g: (np.random.uniform()<g),gaussian))
    cleanVals = list(map(lambda d,b: (d>b),r,boundaryR))
    dirtyVals = list(map(lambda v,f: v^f, cleanVals,flip))
    points = list(map(lambda i,d,v: [i,d,v],x,y,dirtyVals))
    return points

#plotting sample set

def main ():
    tdata = np.array(getPointsConic(coVec, 100, 0, 0, -10, 10, -10, 10))
    vdata = np.array(getPointsConic(coVec, 100, 0, 0, -10, 10, -10, 10))

    plotData(tdata)
    plotData(vdata)
