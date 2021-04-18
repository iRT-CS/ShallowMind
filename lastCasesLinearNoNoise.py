from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
from db import createDatasetsDocument, createNeuralNetsDocument, createExperimentsDocument
import Datasets.GaussianBoundary as gb
import numpy as np
import keras
from testNN import test
from Utils import iterate,plotData
from generateNN import make
import matplotlib.pyplot as plt
from numpy import genfromtxt

# Continuously runs epochs on neural net with given data points until error is minimized
# nn: compiled neural net
# tdata = training data
# vdata = validation data

# coVec = [1,0]

# "/Users/RheaMacBook/Downloads/datasets2.csv"

# later - explicitly create 10 datasets, for each dataset, create & test all neural nets

coVec = [1,0]

#peak, sigma
# peak - max probability of miscategorizing a point, sigma - band of miscategorized points
# raise the sigma - noise points can get farther and farther away from the boundary line
# raise the peak - concentration of miscategorized points goes up
# noiseDist = [0.05, 0.2]
# #peak, sigma
noiseDist = [0,0]
# generate double the points in tdata  (validation split is later), vdata is just a placeholder
tdata1 = genfromtxt('/Users/RheaMacBook/Downloads/datasets2.csv', delimiter=',')
tdata2 = np.array(gb.getPoints(coVec, 2000, noiseDist[0], noiseDist[1], -10, 10, -10, 10))
vdata = np.array(gb.getPoints(coVec, 2000, noiseDist[0], noiseDist[1], -10, 10, -10, 10))

print(tdata1)
print(tdata2)
