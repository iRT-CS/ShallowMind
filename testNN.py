from keras.models import Sequential
from keras.layers import Dense, Activation
from db import createDatasetsDocument, createNeuralNetsDocument, createExperimentsDocument
import GaussianBoundary as gb
import numpy as np
import keras
from Utils import iterate

# Continuously runs epochs on neural net with given data points until error is minimized
# nn: compiled neural net
# tdata = training data
# vdata = validation data
#
def test(nn, tdata, vdata):
    vErrorConsec = 0
    cont = True
    tError = []
    vError = []
    tAcc = []
    vAcc = [];
    epoch = 0
    lowestVError = 1
    statsAtLowestVError = []
    # stopping criterion
    stopC = {
        "Every 5 epochs":[],
        "Validation error increases for 5 consec epochs":[],
        "Validation error increases for 10 consec epochs":[],
        "Validation error increases for 15 consec epochs":[],
        "Decrease in training error from 1 epoch to next is below %1":[],
        "Training error below 15%":[],
        "Training error below 10%":[],
        "Training error below 5%":[],
        "Lowest validation error":[],
    }
    while(cont):
        #train (1) epochs
        nn.fit(x=tdata, y=vdata, batch_size=100, epochs=1, verbose=1)
        # call evaluate - record test & validation error
        stats = model.evaluate(x=training, y=testing, batch_size=100, epochs=1, verbose=1)
        epoch += 1
        # record test error & accuracy
        tError.append(stats[0])
        tAcc.append(stats[1])
        # record validation error & accuracy
        vError.append(stats[2])
        vAcc.append(stats[3])
        # final training error, final validation error, final weights if needed for stopC
        # get_weights returns a list of numpy arrays
        finalStats = [stats[0], stats[2], nn.get_weights()]
        if( vError[len(vError)-1] < lowestVError ):
            lowestVError = vError[len(vError)-1]
            statsAtLowestVError = finalStats
        if( epoch % 5 == 0 ):
            stopC["Every 5 epochs"].append(finalStats)
        # if validation error this epoch increases from val error from the previous epoch
        if(len(vError) > 1 and vError[len(vError)-1] > vError[len(vError)-2]):
            vErrorConsec += 1
        if(vErrorConsec > 5):
            stopC["Validation error increases for 5 consec epochs"].append(finalStats)
        if(vErrorConsec > 10):
            stopC["Validation error increases for 10 consec epochs"].append(finalStats)
        if(vErrorConsec > 15):
            stopC["Validation error increases for 15 consec epochs"].append(finalStats)
        if( tError[len(tError)-1] < 0.15 ):
            stopC["Training error below 15%"].append(finalStats)
        if( tError[len(tError)-1] < 0.10 ):
            stopC["Training error below 10%"].append(finalStats)
        if( tError[len(tError)-1] < 0.05 ):
            stopC["Training error below 5%"].append(finalStats)
        if(len(vError) > 1 and ( vError[len(vError)-2] - vError[len(vError)-1] ) < 0.01 ):
            stopC["Decrease in training error from 1 epoch to next is below %1"].append(finalStats)
    stopC["Lowest validation error"] = statsAtLowestVError
    return tAcc, vAcc, stopC

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
coVec = gb.genFunctionUniform()
tdata = np.array( gb.getPoints(coVec, 1000, 3, 7, -100, 100, -100, 100) )
vdata = np.array( gb.getPoints(coVec, 1000, 3, 7, -100, 100, -100, 100) )
createDatasetsDocument(coVec, [3, 7], [-100, 100, -100, 100], tdata, vdata)

# iterates through all ids and creates neural nets
nets = []
for struct in ids:
    layers = []
    for i in struct:
        layers.append(int(i))
    # the shape wasn't working, so I took out the list dependency
    nets.append(make(IN_SHAPE, layers, OUT_SHAPE, 1, 'tanh'))

# runs test for each neural net
for index,nn in enumerate(nets):
    # Colin did something with layer sizes above - what is that? Is below just repeating code?
    layerSizes = []
    for l in nn.layers:
        layerSizes.append(l.get_output_at(0).get_shape())
    createNeuralNetsDocument(layerSizes, IN_SHAPE, OUT_SHAPE, nn.get_weights(), 'glorot', 'sigmoid')
    # what is the dataset ID? for now, I'm just setting it to 1
    tAcc, vAcc, stoppingCriterionDictionary = test(nn, tdata, vdata)
    createExperimentsDocument(ids[index], layerSizes, IN_SHAPE, OUT_SHAPE, 1, tAcc, vAcc, stoppingCriterionDictionary)
