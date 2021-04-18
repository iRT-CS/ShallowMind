from Visualizations import Visualizations
from VisualizationModel import VisualizationModel
from numpy.random import seed
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import keras
import generateNN
import datetime
import time
import Datasets.GaussianBoundary as gb
from Utils.utils import seeding
import os
import math
from pathlib import Path

class VisualizationCallbacks(tf.keras.callbacks.Callback):

    def __init__(self, model_name, exp_num, dataset_name):
        self.model_name = model_name
        self.exp_num = exp_num
        self.dataset_name = dataset_name
        self.save_path = f".local\\visualizations\\exp-{self.exp_num}\\{self.dataset_name}\\{self.model_name}\\sequence\\raw"
        self.visualizer = Visualizations()

    def on_epoch_end(self, epoch, logs=None):
        _ = 0
