from numpy.random import seed
import tensorflow as tf
import numpy as np
import generateNN
import datetime
import time
from Utils import seeding
import os
import math
from pathlib import Path
from VisCallbacks import VisualizationCallbacks
import Datasets.DatasetGenerator as dg
# import runExp

class VisualizationModel():
    def __init__(self, exp_num, seed=1):
        seed = seed
        seeding.setSeed(seed)
        np.random.seed(seed) # fix random seed
        tf.random.set_seed(seed)

        self.exp_num = exp_num
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        self.modelFolder = f".local\\models\\exp-{self.exp_num}"
        

    def createNetwork(self, hidden_layer_shape):
        self.layer_shape = hidden_layer_shape # shape of hidden layers
        # too lazy to install pymongo, importing these from runExp isnt working so here
        MAX_NODES = 6
        MAX_LAYERS = 4

        IN_SHAPE = (2,)
        OUT_SHAPE = (1,)

        NODES_INLAYER = 2
        NODES_OUTLAYER = 1
        input_num = NODES_INLAYER # x,y
        input_shape = IN_SHAPE # x,y shape
        input_act = "tanh"
        # hidden_act = "sigmoid"
        output_shape = NODES_OUTLAYER
        seed = seeding.getSeed()
        model = generateNN.make(
            input=input_num,
            vec=hidden_layer_shape,
            out=output_shape, 
            shape=input_shape,
            act=input_act,
            seedNum=seed)
        return model
    


    def trainNetwork(self, model, epochs, training_data, validation_data):
        self.modelId = f"{self.getModelId(self.overwrite)}-{self.layer_shape}"
        dateStr = datetime.datetime.now().strftime("%d-%mT%H-%M")
        logdir = os.path.join(".local\\logs", f"exp-{self.exp_num}\\{self.modelId}_{dateStr}")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
        # create early stopping metric
        # using loss rn, maybe switch to accuracy
        # also adjust min_delta, and consider restoring best weights
        early_stop_callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_acc",
            patience=30,
            restore_best_weights=False)
        
        # create model checkpoint callback
        checkpoint_dir = f"{self.modelFolder}\\model-{self.modelId}\\checkpoints" + "\\ep-{epoch}_vAcc-{val_acc:.3f}"
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_dir,
            save_weights_only=False,
            monitor='val_acc',
            save_best_only=True)

        visualizer = VisualizationCallbacks(self.modelId, self.exp_num, self.datasetId)

        tCoords, tLabels = training_data
        # start timer
        start_time = time.time()
        # f i t
        model.fit(
        x=tCoords,
        y=tLabels,
        validation_data=validation_data,
        epochs=epochs,
        callbacks=[
            early_stop_callback,
            tensorboard_callback,
            checkpoint_callback])
        
        self.saveModel(model)

        end_time = time.time()
        self.evaluateNetwork(model, validation_data)
        time_elapsed = end_time - start_time
        timeStr = self.time_convert(time_elapsed)
        print(f"Completed training in {timeStr}")
    
    def time_convert(self, sec):
        mins = math.floor(sec // 60)
        sec = sec % 60
        hours = math.floor(mins // 60)
        mins = mins % 60
        return f"{hours}h, {mins}m, {sec:.2f}s"
    
    def saveModel(self, model):
        # try to save to exp-#/model-####/model
        try:
            indexStr = self.modelId
            new_folder = f"{self.modelFolder}\\model-{indexStr}\\model"
            # make directory for the model folder
            Path(new_folder).mkdir(parents=True, exist_ok=True)

            model.save(new_folder)
            print(f"Model saved successfully to: {new_folder}")
        # if it breaks, overwrite whatevers in save-error-backup with the model
        except:
            backup = f".local\\models\\save-error-backup"
            Path(backup).mkdir(parents=True, exist_ok=True)
            model.save(backup)
            print(f"Error occured while saving, saved to backup at: {backup}")
            # raise error AFTER it saved
            raise

    def evaluateNetwork(self, model, data, isVerbose=True):
        verbose = 1 if isVerbose else 0
        coords, labels = data
        score = model.evaluate(x=coords, y=labels, verbose=verbose)
        print()
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

    def getModelId(self, getCurrent=False):
        add = 1 if getCurrent is False else 0
        # if theres no exp folder, make it
        if os.path.isdir(self.modelFolder) is False:
            os.mkdir(self.modelFolder)
        # look at the last file in the exp folder and extract the id
        # if theres none, its '0000'
        folderList = os.listdir(self.modelFolder)
        indexStr = ""
        if len(folderList) == 0:
            indexStr = "0000"
        else:
            latestFolder = folderList[len(folderList)-1]
            index = int(latestFolder[6:10])
            indexStr = str(index+add).zfill(4)
        return indexStr
    
    def trainNewNetwork(self, epochs, shape, dataset_options, model=None, overwrite=False):
        self.overwrite = overwrite
        # training_data = self.generateTrainingDataset()
        # validation_data = self.generateValidationDataset()
        seed = seeding.getSeed()
        training_data = dg.getDataset(dataset_options.name, dataset_options)
        val_options = dataset_options
        val_options.seed = dataset_options.seed * 2
        validation_data = dg.getDataset(val_options.name, val_options)

        model = self.createNetwork(shape) if model is None else model
        print(model.summary())
        self.trainNetwork(model=model, training_data=training_data, validation_data=validation_data, epochs=epochs)
