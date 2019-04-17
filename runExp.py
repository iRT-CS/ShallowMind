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

# coVec = [1,0]

# later - explicitly create 10 datasets, for each dataset, create & test all neural nets

coVec = [1,0]

tdata = np.array(gb.getPoints(coVec, 1000, 0, 0, -10, 10, -10, 10))
vdata = np.array(gb.getPoints(coVec, 1000, 0, 0, -10, 10, -10, 10))

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

datasetID = createDatasetsDocument(coVec, [3, 7], [-100, 100, -100, 100], tdata.tolist(), vdata.tolist()) # in the first list is peak & sigma, second list is the bounds for the data generation piece

iter = [1]
while(iter != -1):
    print("iter = " + str(iter))
    actualNet = make(NODES_INLAYER, iter, NODES_OUTLAYER, IN_SHAPE, 'tanh')
    weights = list(map(np.ndarray.tolist, actualNet.get_weights()))
    nnID = createNeuralNetsDocument(iter, IN_SHAPE, OUT_SHAPE, weights, 'glorot', 'sigmoid')

    tAcc, vAcc, stoppingCriterionDictionary = test(actualNet, tdata, vdata)
    createExperimentsDocument(nnID, iter, IN_SHAPE, OUT_SHAPE, datasetID, tAcc, vAcc, stoppingCriterionDictionary)

    iter = iterate(iter,MAX_LAYERS,MAX_NODES)
