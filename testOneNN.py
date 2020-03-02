from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
from db import createDatasetsDocument, createNeuralNetsDocument, createExperimentsDocument
import GaussianBoundary as gb
import numpy as np
import keras
from testNN import test
from Utils import iterate,plotData
from generateNN import make
import matplotlib.pyplot as plt

# Continuously runs epochs on neural net with given data points until error is minimized
# nn: compiled neural net
# tdata = training data
# vdata = validation data

coVec = [1,0]

# later - explicitly create 10 datasets, for each dataset, create & test all neural nets

# coVec = [0.125, 0, -4] # 0.125x^2 + 0x -4

# peak - max probability of miscategorizing a point, sigma - band of miscategorized points
noiseDist = [0, 0]
tdata = np.array(gb.getPoints(coVec, 2000, noiseDist[0], noiseDist[1], -10, 10, -10, 10))
vdata = np.array(gb.getPoints(coVec, 2000, noiseDist[0], noiseDist[1], -10, 10, -10, 10))
datasetID = createDatasetsDocument(coVec, noiseDist, [-10, 10, -10, 10], tdata, vdata)
# in the first list is peak & sigma, second list is the bounds for the data generation piece
