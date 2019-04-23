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
#from testNN import test
def test(nn, tdata, vdata):
    tCoords = tdata[:, [0,1]]
    tLabels = tdata[:, [2]]
    #tLabels = to_categorical(tLabels, num_classes=None)

    # print(tLabels)

    vCoords = vdata[:, [0,1]]
    vLabels = tdata[:, [2]]

    vErrorConsec = 0
    cont = True
    tError = []
    vError = []
    tAcc = []
    vAcc = [];
    epoch = 0
    lowestVError = 1
    statsAtLowestVError = []
    flatline = 0
    # stopping criterion
    stopC = {
        "Every 5 epochs":[],
        "Validation error increases for 5 consec epochs":[], #0
        "Validation error increases for 10 consec epochs":[], #1
        "Validation error increases for 15 consec epochs":[], #2
        "Decrease in training error from 1 epoch to next is below %1":[], #3
        "Training error below 15%":[], #4
        "Training error below 10%":[], #5
        "Training error below 5%":[], #6
        "Lowest validation error":[] #7
    }
    #indicates if stopping criterion has been completed
    comp = [False, False, False, False, False, False, False, False]

    #while(any(not value for value in comp)):
    while(not (comp[7] and (comp[2]or flatline> 10))):
        #train (1) epochs
        nn.fit(x=tCoords, y=tLabels, batch_size=100, epochs=1, verbose=0)
        # call evaluate - record test & validation error
        stats = nn.evaluate(x=vCoords, y=vLabels, batch_size=100, verbose=0)
        epoch += 1

        # print(stats)

        # record training error & accuracy
        tError.append(stats[0]) # training error
        tAcc.append(stats[1]) # training accuracy
        # record validation error & accuracy
        vError.append(stats[0]) # validation error
        vAcc.append(stats[1]) # validation accuracy

        # final training error, final validation error, final weights if needed for stopC
        # get_weights returns a list of numpy arrays - convert to arrays using map and tolist

        finalStats = {
            "Final validation error":stats[1],
            "Final training error":stats[0], #0
            "Final weights":list(map(np.ndarray.tolist, nn.get_weights())) #1
        }

        # finalStats = [stats[0], stats[2], nn.get_weights()]
        if( vError[len(vError)-1] < lowestVError ):
            lowestVError = vError[len(vError)-1]
            statsAtLowestVError = finalStats
        if( epoch % 5 == 0 ):
            stopC["Every 5 epochs"].append(finalStats)
        # if validation error this epoch increases from val error from the previous epoch
        if(len(vError) > 1 and vError[len(vError)-1] > vError[len(vError)-2] and not comp[0]):
            vErrorConsec += 1
            comp[0] = True
            print(0)
        if(vErrorConsec > 5 and not comp[1]):
            stopC["Validation error increases for 5 consec epochs"].append(finalStats)
            comp[1] = True
            print(1)
        if(vErrorConsec > 10 and not comp[2]):
            stopC["Validation error increases for 10 consec epochs"].append(finalStats)
            comp[2] = True
            print(2)
        if(vErrorConsec > 15 and not comp[3]):
            stopC["Validation error increases for 15 consec epochs"].append(finalStats)
            comp[3] = True
            print(3)
        if( tError[len(tError)-1] < 0.15 and not comp[4]):
            stopC["Training error below 15%"].append(finalStats)
            comp[4] = True
            print(4)
        if( tError[len(tError)-1] < 0.10 and not comp[5]):
            stopC["Training error below 10%"].append(finalStats)
            comp[5] = True
            print(5)
        if( tError[len(tError)-1] < 0.05 and not comp[6]):
            stopC["Training error below 5%"].append(finalStats)
            comp[6] = True
            print(6)
        if(len(vError) > 1 and ( vError[len(vError)-2] - vError[len(vError)-1] ) < 0.01 ):
            stopC["Decrease in training error from 1 epoch to next is below %1"].append(finalStats)
            comp[7] = True
            flatline += 1
            print(7)
        else:
            flatline = 0
    stopC["Lowest validation error"] = statsAtLowestVError
    return tAcc, vAcc, stopC

coVec = [1,0]

tdata = np.array( gb.getPoints(coVec, 1000, 0, 0, -10, 10, -10, 10) )
vdata = np.array( gb.getPoints(coVec, 1000, 0, 0, -10, 10, -10, 10) )

datasetID = createDatasetsDocument(coVec, [0, 0], [-100, 100, -100, 100], tdata.tolist(), vdata.tolist())

MAX_NODES = 6
MAX_LAYERS = 4

IN_SHAPE = (2,)
OUT_SHAPE = (1,)

NODES_INLAYER = 2
NODES_OUTLAYER = 1

layers = iterate([], MAX_LAYERS, MAX_NODES)

# print(layers)

test1nn = make(NODES_INLAYER, layers, NODES_OUTLAYER, IN_SHAPE, 'tanh')

weights = list(map(np.ndarray.tolist, test1nn.get_weights()))
nnID = createNeuralNetsDocument(layers, IN_SHAPE, OUT_SHAPE, weights, 'glorot', 'sigmoid')

tAcc, vAcc, stoppingCriterionDictionary = test(test1nn, tdata, vdata)
createExperimentsDocument(nnID, layers, IN_SHAPE, OUT_SHAPE, datasetID, tAcc, vAcc, stoppingCriterionDictionary)
