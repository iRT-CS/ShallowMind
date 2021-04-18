
from numpy.random import seed
import tensorflow as tf
import numpy as np
from Utils import seeding
import os
import math
from pathlib import Path
import Vis

class VisualizationCallbacks(tf.keras.callbacks.Callback):

    def __init__(self, model_name, exp_num, dataset_name):
        self.model_name = model_name
        self.exp_num = exp_num
        self.dataset_name = dataset_name
        self.save_path = f".local\\visualizations\\exp-{self.exp_num}\\{self.dataset_name}\\{self.model_name}\\sequence\\raw"

    def on_epoch_end(self, epoch, logs=None):
        _ = 0
