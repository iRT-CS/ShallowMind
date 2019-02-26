from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
from db import createDatasetsDocument, createNeuralNetsDocument, createExperimentsDocument
import GaussianBoundary as gb
import numpy as np
import keras
from Utils import iterate,plotData
from generateNN import make
import matplotlib.pyplot as plt
from testNN import test

tdata = np.array( gb.getPoints(coVec, 1000, 0, 0, -10, 10, -10, 10) )
vdata = np.array( gb.getPoints(coVec, 1000, 0, 0, -10, 10, -10, 10) )

coVec = [1,0]

createDatasetsDocument(coVec, [0, 0], [-100, 100, -100, 100], tdata.tolist(), vdata.tolist())

MAX_NODES = 6
MAX_LAYERS = 4

IN_SHAPE = (2,)
OUT_SHAPE = (1,)

NODES_INLAYER = 2
NODES_OUTLAYER = 1

test1nn = make(NODES_INLAYER, layers, NODES_OUTLAYER, IN_SHAPE, 'tanh')

weights = list(map(np.ndarray.tolist, test1nn.get_weights()))
createNeuralNetsDocument(layers, IN_SHAPE, OUT_SHAPE, weights, 'glorot', 'sigmoid')

tAcc, vAcc, stoppingCriterionDictionary = test(nn, tdata, vdata)
createExperimentsDocument(ids[index], layers, IN_SHAPE, OUT_SHAPE, 1, tAcc, vAcc, stoppingCriterionDictionary)
