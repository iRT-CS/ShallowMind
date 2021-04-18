from tensorflow.python.keras.backend import switch
from VisualizationModel import VisualizationModel
from numpy.random import seed as np_seed
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import keras
import generateNN
import datetime
import time
import Datasets.GaussianBoundary as gb
from Utils import seeding
import os
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import Utils.VisualizationColormaps as vcmap
from Datasets import DatasetGenerator as dg
import Visualizations as vis


"""Gets visualizations for a list of models
:param model_list: the list of models to use
:param dataset_options: ds.DataTypes(.options) - the options for the dataset
"""
def runModelBatch(model_list, dataset_options):
    plotDataset=True
    for id in model_list:
        model_path = MODEL_LOAD_PATH.format(exp_num, id)
        save_path = VIS_SAVE_PATH.format(exp_num, seed, id)
        model = tf.keras.models.load_model(model_path)
        vis.getVisualizations(model=model, dataset_options=dataset_options, save_path=save_path, name=id, plotDataset=plotDataset)
        plotDataset=False

"""Gets visualizations for checkpoints within a list of models
Use useAuto and optionally step_percent, OR checkpoint_list

:param model_list: list - the list of models to get checkpoints from
:param dataset_options: dg.DataTypes(.options) - the options for the dataset
:param checkpoint_list: py.list - the checkpoints to use for visualizations
:param useAuto: boolean - whether to automatically generate the checkpoints or use the checkpoint_list
:param step_percent: float - if using auto, the percent of the checkpoints to get
(.1 gets 10% of the checkpoints, in a 100 checkpoint list itll get each tenth one + the first and last)
"""
def runCheckpointBatch(model_list:list, dataset_options:dg.DataTypes, checkpoint_list:bool=None, useAuto:bool=True, step_percent:float=0.1):
    plotDataset=True
    for model_id in model_list:
        checkpoint_list = getCheckpoints(model_path=model_path, step_percent=step_percent) if useAuto else checkpoint_list
        for checkpoint in checkpoint_list:
            model_path = CHECKPOINT_LOAD_PATH.format(exp_num, model_id, checkpoint)
            model = tf.keras.models.load_model(model_path)
            save_path = VIS_SAVE_PATH.format(exp_num, seed, model_id)
            vis.getVisualizations(model=model, dataset_options=dataset_options, save_path=save_path, name=checkpoint, plotDataset=plotDataset)
            plotDataset=False

"""Gets checkpoints from the checkpoint folder of the given model path

:param mode
"""
def getCheckpoints(model_path, step_percent=0.2):
    all_checkpoints = sorted(os.listdir(model_path), key=lambda dir: dir[-5:])
    checkpoint_list = []
    length = len(all_checkpoints)
    step = round(length * step_percent)
    checkpoint_list += all_checkpoints[:length-1:step]
    checkpoint_list.append(all_checkpoints[length-1])
    return checkpoint_list

def createNetworkSet(layer_shape_list):
    for shape in layer_shape_list:
        network = VisualizationModel(exp_num, seed)
        network.trainNewNetwork(epochs=epochs, shape=shape, dataset_options=dataset_options)
        

MODEL_LOAD_PATH = ".local\\models\\exp-{exp_num}\\model-{id}\\model"
CHECKPOINT_LOAD_PATH = ".local\\models\\exp-{exp_num}\\model-{model_id}\\checkpoints\\{checkpoint}"
VIS_SAVE_PATH = ".local\\visualizations\\exp-{exp_num}\\seed-{seed}\\model-{model_id}"

dataset_options = dg.DataTypes.GaussianBoundaryOptions()
exp_num = 2
seed = 1
epochs = 4