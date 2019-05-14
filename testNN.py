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

class MonitorNN(keras.callbacks.Callback):

    #Log function
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
        createExperimentsDocument(self.nnid, self.struct, self.inshape, self.outshape, self.dsid, self.losses, self.val_losses, self.stoppingCriterionDictionary)

    def __init__(self, nnid, struct, inshape, outshape, dsid):
        self.nnid = nnid
        self.struct = struct
        self.inshape = inshape
        self.outshape = outshape
        self.dsid = dsid

    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_loss_count = 0
        self.lowest_val_acc = float('inf')
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

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))

        #Epoch-wise log
        if epoch % 5 == 0:
            self.log("Every 5 epochs")

        #Increment counter if validation loss increases
        if (len(self.val_losses) >= 2) and (self.val_losses[-1] - self.val_losses[-2] > 0):
            self.val_loss_count += 1
        else:
            self.val_loss_count = 0

        if self.val_loss_count == 5:
            self.log("Validation error increases for 5 consec epochs")
        if self.val_loss_count == 10:
            self.log("Validation error increases for 10 consec epochs")
        if self.val_loss_count == 15:
            self.log("Validation error increases for 15 consec epochs")

        #Hard stop for slow validation error improvement
        if (len(self.losses) >= 5) and (self.losses[-2] - self.losses[-1] < 0.001):
            print('Ended (slow improvement)')
            self.end()

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

        #Hard training stop
        if epoch == 2000:
            print('Ended (2000 epochs complete)')
            self.log("Lowest validation error")
            self.end()


earlyStoppingLoss = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=0, verbose=0, mode='min', baseline=None, restore_best_weights=False)


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

    lowestVError = 1
    statsAtLowestVError = []
    flatline = 0

    #workaround for bug asssociated with validation_data argument
    #TODO figure out why we can't train with validation_data
    tCoords = np.concatenate(tCoords, vCoords)
    tLabels = np.concatenate(tLabels, vLabels)

    monitor = MonitorNN(nnid, struct, inshape, outshape, dsid)
    nn.fit(x=tCoords, y=tLabels, batch_size=100, epochs=2000, verbose=1, callbacks=[monitor], validation_split=0.5)
    print(monitor.stoppingCriterionDictionary)
