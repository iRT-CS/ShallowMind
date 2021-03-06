from numpy.random import seed
import tensorflow
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import keras


'''
Testing Procedure:

-Generate 1000 training points
-Generate 1000 test points
-For each step record training and test accuracy
-Stop at ?

-Record the number of steps
-Record NN A. function
-Record NN
-Record sigmoid, peak (noise distribution) polynomial
'''

# make neural net:
# in: number of nodes in input layer
# vec: numbers of nodes in hidden layers
# out: number of nodes in output layer
# shape: shape of input data
# act: activation function ('sigmoid' or 'tanh' most likely)

def make(input, vec, out, shape, act, seedNum):
    seed(seedNum)
    tensorflow.random.set_seed(seedNum)
    
    net = Sequential()
    net.add(Dense(input, input_shape = shape, activation = act))
    for i in vec:
        net.add(Dense(i, activation = act))
    net.add(Dense(out, activation = 'sigmoid')) # why does it change from tanh to sigmoid here
    net.compile(loss='mse',
              optimizer='sgd',
              metrics=['acc'])
    return net
