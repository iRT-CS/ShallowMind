
from matplotlib.pyplot import plot
from numpy.lib.npyio import save
from numpy.random import seed
import tensorflow as tf
import numpy as np
from Utils import seeding
import os
import math
from pathlib import Path
import Vis
import Datasets.DatasetGenerator as dg

"""Class for creating visualizations during training
"""
class VisualizationCallbacks(tf.keras.callbacks.Callback):

    """Initialize the callback

    :param model_name: str - the id of the model (either just id (####) or id with structure (####-[#,#]))
    :param exp_num: int - the experiment number
    :param dataset_options: ds.Datatypes(.options) - the options for the dataset
    :param target_epochs: the max epochs to train the model for
    """
    def __init__(self, model_name:str, exp_num:int, dataset_options:dg.DataTypes, target_epochs:int):
        self.model_name = model_name
        self.exp_num = exp_num
        self.dataset_options = dataset_options
        self.dataset = dg.getDataset(self.dataset_options.name, self.dataset_options)
        # zfill so it sorts alphabetically in folders
        self.epoch_zFill = len(str(target_epochs))
        self.save_path = f".local\\visualizations\\exp-{self.exp_num}\\{self.dataset_options.name}\\{self.model_name}\\sequence"
        self.save_path_raw = f"{self.save_path}\\raw"
        self.lastEpoch = 0
        # self.data_array=[]
        self.graphDataset()

    """Graphs the dataset to the sequence folder
    """
    def graphDataset(self):
        dataset = dg.getDataset(self.dataset_options.name, self.dataset_options)
        plot_name = f"Original Dataset | Dataset: {self.dataset_options.name}"
        Vis.graphDataset(
            dataset=dataset,
            save_path=self.save_path_raw,
            plot_name = plot_name,
            dataset_options=self.dataset_options)

    """Action to be performed when epoch ends, called automatically by tensorflow<3
    For us, it uses the model at this state and makes a prediction visualization
    which is saved into the sequence folder to later be turned into a gif

    :param epoch: int - the current epoch
    :param logs: dict - logs containing metrics for the model
    """
    def on_epoch_end(self, epoch:int, logs:dict=None):
        epochStr = str(epoch).zfill(self.epoch_zFill)
        acc = logs["val_acc"]
        filename=f"ep-{epochStr}_vAcc-{acc:.2f}"
        plot_name = f"Epoch: {epochStr} | Accuracy: {(acc*100):.2f}% | Dataset: {self.dataset_options.name}"
        # data_arr = [self.model, filename, plot_name]
        # self.data_array.append(data_arr)
        self.graphModelPredictions(filename, plot_name)
        self.lastEpoch = epoch

    """Function that gets called when the training ends
    basically plots the final predictions, then creates a gif out of the sequences

    :param logs: dict - logs containing metrics for the model
    """
    def on_train_end(self, logs:dict=None):
        # graph final model predictions
        acc = logs["val_acc"]
        filename = f"final_eps-{self.lastEpoch+1}_vAcc-{acc:.2f}"
        plot_name = f"Final | Epochs: {self.lastEpoch+1} | Accuracy: {(acc*100):.2f}% | Dataset: {self.dataset_options.name}"
        self.graphModelPredictions(filename=filename, plot_name=plot_name)
        
        # make file name and save path for full gif
        file_name = f"sequence_exp-{self.exp_num}_{self.model_name}.gif"
        save_path = f"{self.save_path}\\{file_name}"

        # so the gif creator thing can take in an array with times to display each frame,
        # so this creates that list and makes the first and last one display for 3 seconds
        # and the rest for 0.5
        plot_list = os.listdir(self.save_path_raw)
        duration_arr = np.full(shape=len(plot_list), fill_value=0.5)
        duration_arr[[0, len(plot_list)-1]] = 3
        duration_list = list(duration_arr)
        # make it
        Vis.createSequence(image_path=self.save_path_raw, save_path=save_path, duration_list=duration_list)

    """Graphs predictions for the model
    saves to .local

    :param filename: str - the filename to save the model as
    :param plot_name: str - the title of the plot
    """
    def graphModelPredictions(self, filename:str, plot_name:str):
        Vis.graphPredictions(
            dataset=self.dataset,
            model=self.model,
            save_path=self.save_path_raw,
            name=filename,
            plot_name=plot_name,
            dataset_options=self.dataset_options)