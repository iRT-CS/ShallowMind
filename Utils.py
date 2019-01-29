import math
from db import createDatasetsDocument
from GaussianBoundary import getPoints
import matplotlib.pyplot as plt
def iterate(nnID,maxLayers,maxNodes):
     curLayer = 0
     while(curLayer<len(nnID)):
          if(nnID[curLayer] < maxNodes):
               nnID[curLayer] += 1
               return nnID
          else:
               curLayer += 1
     if(curLayer >= maxLayers):
          return -1
     else:
          nnID = [1]*(curLayer+1)
          return nnID
     '''
    #get digits of the function
    #find number of digits
     numDigits = math.log10(nnID)
    #Because 10 has 2 digits
     if(numDigits%1 == 0):
         numDigits = int(numDigits + 1)
     else:
         numDigits = int(math.ceil(numDigits))
    #get digits into array(backwards)
     digits = []
     for i in range(numDigits):
         digit = (nnID//(10**i))%10
         digits.append(digit)
     digits.append(0)
    #Find carryover
     digitToIncrease = 0
     while((digitToIncrease<maxLayers)and(digits[digitToIncrease]>=maxNodes)):
         digitToIncrease = digitToIncrease + 1
     if(digitToIncrease == maxLayers):
         return -1
     for i in range(digitToIncrease):
         digits[i] = 1
     digits[digitToIncrease] = digits[digitToIncrease] + 1
     #recombine into a number
     newID = 0
     for i in range(len(digits)):
         newID += (10**i)*digits[i]
     return newID
     '''
def addDataset(polynomial,noiseDistribution,dataRange):
     peak = noiseDistribution[0]
     sigma = noiseDistribution[1]
     xMin= dataRange[0]
     xMax = dataRange[1]
     yMin = dataRange[2]
     yMax = dataRange[3]
     trainingValues = getPoints(polynomial,1000,sigma,peak,xMin,xMax,yMin,yMax)
     testValues = getPoints(polynomial,1000,sigma,peak,xMin,xMax,yMin,yMax)
     createDataSetsDocument(polynomial,noiseDistribution,dataRange,trainingValues,testValues)

def plotData(data):
    xs = [pt[0] for pt in data]
    ys = [pt[1] for pt in data]
    # print(data)
    for i in range(0,len(data)):
        # print(data[i])
        if data[i][2] == 1:
            plt.plot(xs[i], ys[i], 'y^')
        else:
            plt.plot(xs[i], ys[i], 'bs')

    plt.show()
