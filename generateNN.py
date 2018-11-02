from keras.models import Sequential
from keras.layers import Dense, Activation

'''
make neural net:
in: number of nodes in input layer
vec: numbers of nodes in hidden layers
out: number of nodes in output layer
shape: shape of input data
act: activation function ('sigmoid' or 'tanh' most likely)
'''

def make(in, vec, out, shape, act):
    net = new Sequential()
    net.add(Dense(in, input_shape = shape, activation = act))
    for i in vec:
        net.add(Dense(i, activation = act))
    net.add(Dense(out, act = act))
    return net
