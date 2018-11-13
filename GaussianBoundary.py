from random import randrange, gauss
import numpy as np
from scipy.optimize import fmin_cobyla
#import timeit
def genFunctionUniform(degree = 2,minimum = -7,maximum = 7):
    coefficients = []
    for i in range(degree+1):
        coefficient = randrange(minimum,maximum)
        coefficients.insert(0,coefficient)
    return coefficients

def genFunctionGaussian(degree = 2,mean = 0,sigma = 7, scaling = False):
    coefficients = []
    for i in range(degree+1):
        if(scaling):
            eMean = (1/(i+1))*mean
        else:
            eMean = mean
        coefficient = gauss(eMean,sigma)
        coefficients.insert(0,coefficient)
    return coefficients

def evalFunction(coVec,ipVar):
    boundaryPoint = 0
    for i in range(len(coVec)):
        termVal = pow(ipVar,i)*coVec[i]
        boundaryPoint += termVal
    return boundaryPoint
def pointDistance(x1,y1,x2,y2):
    return np.sqrt(pow((x1-x2),2)+pow((y1-y2),2))

def distanceToCurve(coVec,ipVar,dpVar):
    maxDistance = abs(dpVar - evalFunction(coVec, ipVar))
    xMin = fmin_cobyla(\
        lambda x: pointDistance(ipVar,dpVar,x[0],evalFunction(coVec, x[0]))\
        ,x0 = [ipVar,ipVar],\
        cons = [lambda x: abs(ipVar-maxDistance)],\
        rhoend = 1e-3)
    return pointDistance(ipVar,dpVar,xMin[0], evalFunction(coVec, xMin[0]))

def getGuassianAtPoint()
def categorize(ipVar,dpVar,coVec,noiseRate):
    boundaryPoint = evalFunction(coVec,ipVar)
    rawCategory = (dpVar >= boundaryPoint)
    #Find noise
    distanceToCurve = distanceToCurve(coVec,ipVar,dpVar)
    
    
        

