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

# Keras Callback Function
# on_epoch_end: return training loss, validation loss

# Array of stopping criterion that is logged during training
stopC = {
    "Every 5 epochs":[],
    "Validation error increases for 5 consec epochs":[], #0
    "Validation error increases for 10 consec epochs":[], #1
    "Validation error increases for 15 consec epochs":[], #2
    "Decrease in training error from 1 epoch to next is below %1":[], #3
    "Training error below 15%":[], #4
    "Training error below 10%":[], #5
    "Training error below 5%":[], #6
    # "Lowest validation error":[] #7
}

#A class that lies atop a neural net as it trains and calls functions at certain intervals
class MonitorNN(keras.callbacks.Callback):

    #log function
    def log(self, criterion):
        finalStats = {
            "Final validation error":self.val_losses[-1],
            "Final training error":self.losses[-1], #0
            "Final weights":list(map(np.ndarray.tolist, self.model.get_weights())) #1
        }
        self.stoppingCriterionDictionary[criterion].append(finalStats)

    #Stop function
    def end(self):
        self.model.stop_training = True
        # createExperimentsDocument(self.nnid, self.struct, self.inshape, self.outshape, self.dsid, self.losses, self.val_losses, self.stoppingCriterionDictionary)

    #Variable initialization, called when neural net is created
    def __init__(self, nnid, struct, inshape, outshape, dsid):
        self.nnid = nnid
        self.struct = struct
        self.inshape = inshape
        self.outshape = outshape
        self.dsid = dsid

    #Initialization function that creates a bunch of lists to contain logs at the beginning of training
    def on_train_begin(self, logs={}):
        #Basic logs
        self.losses = []
        self.val_losses = []
        self.acc = []
        #Counter for epochs where loss goes up
        self.val_loss_count = 0
        #Logs the lowest validation accuracy
        self.lowest_val_acc = float('inf')
        #Initializes the stopping criterion dictionary
        self.stoppingCriterionDictionary = {
            "Every 5 epochs":[],
            "Validation error increases for 5 consec epochs":[], #0
            "Validation error increases for 10 consec epochs":[], #1
            "Validation error increases for 15 consec epochs":[], #2
            "Decrease in training error from 1 epoch to next is below %1":[], #3
            "Training error below 15%":[], #4
            "Training error below 10%":[], #5
            "Training error below 5%":[], #6
            "Training error below 1%":[],
            "Lowest validation error":[] #7
        }

    #TODO: implement lowest validation error
    # stop when training error is very low for a while

    #Callback function that run at the end of every epoch
    def on_epoch_end(self, epoch, logs={}):
        #Adding important values to lists in the class
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))

        #Less frequent logger
        if epoch % 5 == 0:
            self.log("Every 5 epochs")

        #Increment counter if validation loss increases
        if (len(self.val_losses) >= 2) and (self.val_losses[-1] - self.val_losses[-2] > 0):
            self.val_loss_count += 1
        else:
            self.val_loss_count = 0

        #Logs if the loss has gone up for many epochs
        if self.val_loss_count == 5:
            self.log("Validation error increases for 5 consec epochs")
        if self.val_loss_count == 10:
            self.log("Validation error increases for 10 consec epochs")
        if self.val_loss_count == 15:
            self.log("Validation error increases for 15 consec epochs")

        # Hard stop for slow validation error improvement
        # if (len(self.losses) >= 5) and (self.losses[-2] - self.losses[-1] < 0.00001):
            # print('Ended (slow improvement)')
            # self.end()

        #Log training error milestones
        if self.losses[-1] < 0.01:
            self.log("Training error below 1%")
        elif self.losses[-1] < 0.05:
            self.log("Training error below 5%")
        elif self.losses[-1] < 0.1:
            self.log("Training error below 1%")
        elif self.losses[-1] < 0.15:
            self.log("Training error below 15%")

        if self.lowest_val_acc >= logs.get('acc'):
            self.lowest_val_acc = logs.get('acc')

        # Hard training stop (for averaging)
        if epoch == 900:
            print('Ended (900 epochs complete)')
            # self.log("Lowest validation error")
            self.end()


#earlyStoppingLoss = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=0, verbose=0, mode='min', baseline=None, restore_best_weights=False)


# Continuously runs epochs on neural net with given data points until error is minimized
# nn: compiled neural net
# tdata = training data
# vdata = validation data

def test(nn, tdata, vdata, nnid, struct, inshape, outshape, dsid):
    tCoords = tdata[:, [0,1]]
    tLabels = tdata[:, [2]]
    #tLabels = to_categorical(tLabels, num_classes=None)

    # print(tLabels)

    vCoords = vdata[:, [0,1]]
    vLabels = tdata[:, [2]]

    # flatten = lambda l: [for sublist in l sublist[1]]
    # flatten2 = lambda l: [for sublist in l sublist[2]]

    # print(vCoords)
    # print(vLabels)

    # lowestVError = 1
    # statsAtLowestVError = []
    # flatline = 0

    #workaround for bug asssociated with validation_data argument
    #TODO figure out why we can't train with validation_data
    #tCoords = np.concatenate((tCoords, vCoords))
    #tLabels = np.concatenate((tLabels, vLabels))

    # WE DON'T USE VALIDATION DATA - workaround. We just split the training data into 2

    # print(tCoords)

    # coords = np.concatenate((tCoords.T, tLabels.T), axis = 0)
    # coords = coords.T
    # print(coords.shape)

    # plotData(coords)

    monitor = MonitorNN(nnid, struct, inshape, outshape, dsid)
    nn.fit(x=tCoords, y=tLabels, batch_size=100, epochs=2000, verbose=1, callbacks=[monitor], validation_split=.5) #, validation_data = ) #validation_data=(flatten, flatten2))
    # print(monitor.stoppingCriterionDictionary)
    return(monitor.losses, monitor.val_losses, monitor.stoppingCriterionDictionary)
