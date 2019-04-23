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

coVec = [1,0]

tdata = np.array(gb.getPoints(coVec, 1000, 0, 0, -10, 10, -10, 10))
vdata = np.array(gb.getPoints(coVec, 1000, 0, 0, -10, 10, -10, 10))

tCoords = tdata[:, [0,1]]
tLabels = tdata[:, [2]]
    #tLabels = to_categorical(tLabels, num_classes=None)

    # print(tLabels)

vCoords = vdata[:, [0,1]]
vLabels = vdata[:, [2]]

MAX_NODES = 6
MAX_LAYERS = 4

IN_SHAPE = (2,)
OUT_SHAPE = (1,)

NODES_INLAYER = 2
NODES_OUTLAYER = 1

actualNet = Sequential()
actualNet.add(Dense(NODES_INLAYER, input_shape = IN_SHAPE, activation = 'tanh'))
for i in [2,4,2]:
    actualNet.add(Dense(i, activation = 'tanh'))
actualNet.add(Dense(NODES_OUTLAYER, activation = 'tanh'))
actualNet.compile(loss='mse', optimizer='sgd', metrics=['acc'])

actualNet.fit(x=tCoords, y=tLabels, batch_size=100, epochs=500, verbose=2)
actualNet.evaluate(x=vCoords, y=vLabels, batch_size=100, verbose=2)

