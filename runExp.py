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
from seed import getSeed
import matplotlib.pyplot as plt

# Continuously runs epochs on neural net with given data points until error is minimized
# nn: compiled neural net
# tdata = training data
# vdata = validation data

# coVec = [1,0]

# later - explicitly create 10 datasets, for each dataset, create & test all neural nets

coVec = [.25, 0, -5]

# the seed to generate neural networks with - a number that serves as identifier for the experiment (can be used to reproduce results)
seedNum = getSeed()

#peak, sigma
# peak - max probability of miscategorizing a point, sigma - band of miscategorized points
# raise the sigma - noise points can get farther and farther away from the boundary line
# raise the peak - concentration of miscategorized points goes up
# noiseDist = [0.05, 0.2]
# #peak, sigma
noiseDist = [.075, .2]
# generate double the points in tdata  (validation split is later), vdata is just a placeholder
tdata = np.array(gb.getPoints(coVec, 2000, noiseDist[1], noiseDist[0], -10, 10, -10, 10))
vdata = np.array(gb.getPoints(coVec, 2000, noiseDist[1], noiseDist[0], -10, 10, -10, 10)) # vdata is never actually used, just a bug fix. tdata is split in two

# plotting the normal dataset, no noise
# plot the dataset, with noise
# use a parabola, not too wide

# print(tdata)

# plotData(tdata)
# plotData(vdata)

# try all at once for 500 epochs

MAX_NODES = 6
MAX_LAYERS = 4

IN_SHAPE = (2,)
OUT_SHAPE = (1,)

NODES_INLAYER = 2
NODES_OUTLAYER = 1

#vdata is actually never used... fix later with the training split somehow
datasetID = createDatasetsDocument(coVec, noiseDist, [-10, 10, -10, 10], tdata.tolist(), vdata.tolist()) # in the first list is peak & sigma, second list is the bounds for the data generation piece

iter = [1]

# print("iter = " + str(iter))
# actualNet = make(NODES_INLAYER, iter, NODES_OUTLAYER, IN_SHAPE, 'tanh')
# weights = list(map(np.ndarray.tolist, actualNet.get_weights()))
# nnID = createNeuralNetsDocument(iter, IN_SHAPE, OUT_SHAPE, weights, 'glorot', 'sigmoid')

# test(actualNet, tdata, vdata, "12829", iter, IN_SHAPE, OUT_SHAPE, "12829")
# createExperimentsDocument(nnID, iter, IN_SHAPE, OUT_SHAPE, datasetID, tAcc, vAcc, stoppingCriterionDictionary)
# iter = iterate(iter,MAX_LAYERS,MAX_NODES)

while(iter != -1):
    print("iter = " + str(iter))
    actualNet = make(NODES_INLAYER, iter, NODES_OUTLAYER, IN_SHAPE, 'tanh')
    weights = list(map(np.ndarray.tolist, actualNet.get_weights()))
    nnID = createNeuralNetsDocument(iter, IN_SHAPE, OUT_SHAPE, weights, 'glorot', 'sigmoid')
    tAcc, vAcc, stoppingCriterionDictionary = test(actualNet, tdata, vdata, nnID, iter, IN_SHAPE, OUT_SHAPE, datasetID)
    createExperimentsDocument(nnID, iter, IN_SHAPE, OUT_SHAPE, datasetID, tAcc, vAcc, stoppingCriterionDictionary, seedNum)
    # iter = iterate(iter,MAX_LAYERS,MAX_NODES)
    iter = -1
