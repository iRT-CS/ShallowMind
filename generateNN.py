from keras.models import Sequential
from keras.layers import Dense, Activation

'''
make neural net:
in: number of nodes in input layer
vec: numbers of nodes in hidden layers
out: number of nodes in output layer
shape: shape of input data (tuple describing range, with None to show no limit in range)
act: activation function ('sigmoid' or 'tanh' most likely)
TODO?: add initialization arguments 
'''

def make(ins, vec, out, shape, act):
    net = Sequential()
    net.add(Dense(ins, input_shape = shape, activation = act))
    for i in vec:
        net.add(Dense(i, activation = act))
    net.add(Dense(out, activation = act))
    return net

make(1, [1], 1, (0,1), 'tanh')
