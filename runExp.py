from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
from db import createDatasetsDocument, createNeuralNetsDocument, createExperimentsDocument
import GaussianBoundary as gb
import numpy as np
import keras
from testNN.py import test
from Utils import iterate,plotData
from generateNN import make
import matplotlib.pyplot as plt

# Continuously runs epochs on neural net with given data points until error is minimized
# nn: compiled neural net
# tdata = training data
# vdata = validation data

tdata = np.array( gb.getPoints(coVec, 1000, 0, 0, -10, 10, -10, 10) )
vdata = np.array( gb.getPoints(coVec, 1000, 0, 0, -10, 10, -10, 10) )

# plotting the normal dataset, no noise
# plot the dataset, with noise
# use a parabola, not too wide

# print(tdata)

# plotData(tdata)
# plotData(vdata)

datasetID = createDatasetsDocument(coVec, [3, 7], [-100, 100, -100, 100], tdata.tolist(), vdata.tolist()) # in the first list is peak & sigma, second list is the bounds for the data generation piece

# iterates through all ids and creates neural nets
actualNets = []
nnIDs = []
for struct in actualNets:
    # the shape wasn't working, so I took out the list dependency
    actualNets.append(make(NODES_INLAYER, struct, NODES_OUTLAYER, IN_SHAPE, 'tanh'))

    # change the np arrays of weights to lists of lists
    # https://stackoverflow.com/questions/46817085/keras-interpreting-the-output-of-get-weights

    weights = list(map(np.ndarray.tolist, actualNets[len(actualNets)-1].get_weights()))
    nnID = createNeuralNetsDocument(struct, IN_SHAPE, OUT_SHAPE, weights, 'glorot', 'sigmoid')
    nnIDs.append(nnID)

# runs test for each neural net
for index,nn in enumerate(actualNets):
    # what is the dataset ID? for now, I'm just setting it to 1
    tAcc, vAcc, stoppingCriterionDictionary = test(nn, tdata, vdata)
    # print(stoppingCriterionDictionary)
    createExperimentsDocument(nnIDs[index], actualNets[index], IN_SHAPE, OUT_SHAPE, datasetID, tAcc, vAcc, stoppingCriterionDictionary)
