from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import keras
from Utils import iterate

# Continuously runs epochs on neural net with given data points until error is minimized
# nn: compiled neural net
# data: list of data points
#
def test(nn, tdata, vdata):
    training, test = data
    training = np.array(training)
    vErrorConsec = 0
    cont = True
    tError = []
    vError = []

    #boolean consec
    while(cont):
        #train (1) epochs
        nn.fit(x=training, y=test, batch_size=100, epochs=10, verbose=1)

        # call evaluate - record test & validation error
        stats = model.evaluate(x=training, y=testing, batch_size=100, epochs=1, verbose=1)

        # record test error
        tError.append(stats[0])
        # record validation error
        vError.append(stats[2])

        #stopping criterion

        #terminate if error is under 20% and vErrorConsec is greater than 5
        if(len(vError) > 1 and vError[len(vError)-1] < vError[len(vError)-2]):
            vErrorConsec += 1
        if(vError[len(vError)-1] < 0.2 and vErrorConsec > 5):
            break

    return tError, vError,

#
MAX_NODES = 6
MAX_LAYERS = 4

IN_SHAPE = 2
OUT_SHAPE = 1

# create ids in list form
ids = []
id = iterate(1, MAX_LAYERS, MAX_NODES)
newid = 0
while(id != -1):
    ids.append(id)
    newid = iterate(id, MAX_LAYERS, MAX_NODES)
    id = newid

# create data points
coVec = genFunctionUniform()
tdata = getPoints(coVec, 1000, 7, -100, 100, -100, 100)
vdata = getPoints(coVec, 1000, 7, -100, 100, -100, 100)

# iterates through all ids and creates neural nets
nets = []
for struct in ids:
    layers = []
    for i in struct:
        layers.append(int(i))

    nets.append(make(IN_SHAPE, layers, OUT_SHAPE, [,1], 'tanh'))

# runs test for each neural net
for nn in nets:
