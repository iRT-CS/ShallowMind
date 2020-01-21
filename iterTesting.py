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

MAX_NODES = 6
MAX_LAYERS = 4

IN_SHAPE = (2,)
OUT_SHAPE = (1,)

NODES_INLAYER = 2
NODES_OUTLAYER = 1

iter = [1]

count = 0

while(iter != -1):
    print("iter = " + str(iter))
    iter = iterate(iter,MAX_LAYERS,MAX_NODES)
    count = count + 1

print(count)
