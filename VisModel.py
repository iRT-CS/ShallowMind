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
from Utils import VisUtils
# import runExp

class VisualizationModel():
    """Initialize the model instance
    :param exp_num: int - the experiment number for the model
    :param seed: int - the seed number for the randomizations
    """
    def __init__(self, exp_num:int, seed:int=seeding.getSeed()):
        seed = seed
        seeding.setSeed(seed)
        np.random.seed(seed) # fix random seed
        tf.random.set_seed(seed)
        # set base save path for models 
        self.exp_num = exp_num
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        self.modelFolder = f".local\\models\\exp-{self.exp_num}"
        
    """Creates the network with the given specifications

    :param hidden_layer_shape: array-like - the shape of the network, with the
    length of the array as the number of hidden layers and the element as the nodes in that layer
    :returns: tf.keras.model - the model generated
    """
    def createNetwork(self, hidden_layer_shape:list):
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
    

    def createModelCallbacks(self, epochs:int, saveVisualizations:bool, useEarlyStopping:bool):
        """Creates a set of model callbacks and returns them as a list
        
        args:
            epochs: int - the max number of epochs to train the network for
            saveVisualizations: bool - whether to save intermediate visualizations for the network per epoch
            useEarlyStopping: bool - whether to use early stopping in the model
            
            returns: list - the model callbacks
        """

        # list of callbacks
        callbackList = []
        # tensorboard callback, basically stores info about the model as it trains
        # and lets you view it on tensorboard. for info on tensorboard, either
        # look it up or there's a comment on it somewhere (on Vis, might move to here tho)
        dateStr = datetime.datetime.now().strftime("%d-%mT%H-%M")
        logdir = os.path.join(".local\\logs", f"exp-{self.exp_num}\\{self.modelId}_{dateStr}")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
        callbackList.append(tensorboard_callback)
        # create early stopping metric
        # also adjust min_delta, and consider restoring best weights
        # ^ not restoring best weights because its just the last checkpoint saved
        # basically, monitors the supplied metric and stops training if the network hasn't improved this metric
        # after patience epochs
        if useEarlyStopping:
            early_stop_callback = tf.keras.callbacks.EarlyStopping(
                monitor="val_acc",
                patience=15,
                restore_best_weights=False)
            callbackList.append(early_stop_callback)

        # create model checkpoint callback
        # as the network trains, if the validation accuracy improves, it saves the network
        # you can make predictions using these saved models later or even train them more
        checkpoint_dir = f"{self.modelFolder}\\model-{self.modelId}\\checkpoints" + "\\ep-{epoch}_vAcc-{val_acc:.3f}"
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_dir,
            save_weights_only=False,
            monitor='val_acc',
            save_best_only=True)
        callbackList.append(checkpoint_callback)

        # custom callback for saving visualizations as the network trains
        # see class for more info
        if saveVisualizations:
            visualizer_callback = VisualizationCallbacks(
                model_name=self.modelId,
                exp_num=self.exp_num,
                dataset_options=self.dataset_options,
                target_epochs=epochs)
            callbackList.append(visualizer_callback)
        
        return callbackList

    def trainNetwork(self, model:tf.keras.models, epochs:int, training_data:np.ndarray,
        validation_data:np.ndarray, saveVisualizations:bool,  useEarlyStopping:bool):
        """Trains the network

        args:
            model: tf.keras.model - the model to train
            epochs: int - the number of epochs to train for
            training_data: np.ndarray - the training data for the model
            validation_data: np.ndarray - the validation data for the model
            saveVisualizations: bool - whether to save intermediate visualizations for the network per epoch
            useEarlyStopping: bool - whether to use early stopping in the model
        """
        self.modelId = f"{self.getModelId(self.overwrite)}-{self.layer_shape}"
        # get callbacks
        callback_list = self.createModelCallbacks(epochs=epochs, saveVisualizations=saveVisualizations, useEarlyStopping=useEarlyStopping)

        tCoords, tLabels = training_data
        vCoords, vLabels = validation_data
        # start timer
        start_time = time.time()
        # f i t
        model.fit(
        x=tCoords,
        y=tLabels,
        validation_data=(vCoords, vLabels),
        epochs=epochs,
        callbacks=callback_list)
        
        self.saveModel(model)

        end_time = time.time()
        self.evaluateNetwork(model, validation_data)
        time_elapsed = end_time - start_time
        # make time look pretty
        timeStr = VisUtils.time_convert(time_elapsed)
        print(f"Completed training in {timeStr}")
    
    """Saves the model so it can be sent around and loaded again later
    :param model: tf.keras.model - the model to save
    """
    def saveModel(self, model:tf.keras.models):
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
        
    """Evaluates the network and prints scores
    :param model: tf.keras.model - the model to evaluate
    :param data: np.ndarray - the data to evaluate the model against
    :param isVerbose: boolean - whether to show the progress bar or not
    """
    def evaluateNetwork(self, model:tf.keras.models, data:np.ndarray, isVerbose:bool=True):
        verbose = 1 if isVerbose else 0
        coords, labels = data
        score = model.evaluate(x=coords, y=labels, verbose=verbose)
        print()
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

    """Gets the model id for the current model
    Pretty much looks into the model folder and if its set to override,
    it returns the id of the last model in the folder (0000 if none)
    and if its set to get a new one, it adds one to whatever the last one is

    :param getCurrent: boolean - whether to get the last model id or make a new one
    :returns: str - the string of the model id (####)
    """
    def getModelId(self, getCurrent:bool=False):
        add = 1 if getCurrent is False else 0
        # if theres no exp folder, make it
        Path(self.modelFolder).mkdir(parents=True, exist_ok=True)
        # look at the last file in the exp folder and extract the id
        # if theres none, its '0000'
        folderList = os.listdir(self.modelFolder)
        indexStr = ""
        if len(folderList) == 0 or folderList[0].find("model") is -1:
            indexStr = "0000"
        else:
            latestFolder = folderList[len(folderList)-1]
            # this is a really bad setup lol someone should make this loop through all
            if latestFolder.find("model") is -1:
                latestFolder = folderList[len(folderList)-2]
            index = int(latestFolder[6:10])
            indexStr = str(index+add).zfill(4)
        return indexStr
    

    def trainNewNetwork(self, epochs:int, shape:list, dataset_options:dg.DataTypes,
      model:tf.keras.models=None, overwrite:bool=False, saveVisualizations:bool=True, useEarlyStopping:bool=True):
        """Trains a new network with the provided specs

        Args:
            epochs: int - the numebr of epochs to train for
            shape: array_like - the shape of the network
            dataset_options: DataTypes(.options) - the options for the dataset
            model: tf.keras.model - optionally provide a model to train further
            overwrite: boolean - whether to overwrite the last model or save a new one
            saveVisualizations: boolean - whether to save intermediate visualizations
            useEarlyStopping: boolean - whether to use early stopping
        """
        self.dataset_options = dataset_options
        self.overwrite = overwrite
        # training_data = self.generateTrainingDataset()
        # validation_data = self.generateValidationDataset()
        training_data = dg.getDataset(dataset_options.name, dataset_options)
        val_options = dataset_options
        val_options.seed = dataset_options.seed * 2
        validation_data = dg.getDataset(val_options.name, val_options)

        model = self.createNetwork(shape) if model is None else model
        print(model.summary())
        self.trainNetwork(
            model=model, training_data=training_data, validation_data=validation_data,
            epochs=epochs, saveVisualizations=saveVisualizations, useEarlyStopping=useEarlyStopping)
