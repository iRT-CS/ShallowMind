
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

class VisualizationCallbacks(tf.keras.callbacks.Callback):

    def __init__(self, model_name, exp_num, dataset_options, target_epochs):
        self.model_name = model_name
        self.exp_num = exp_num
        self.dataset_options = dataset_options
        self.dataset = dg.getDataset(self.dataset_options.name, self.dataset_options)
        self.epoch_zFill = len(str(target_epochs))
        self.save_path = f".local\\visualizations\\exp-{self.exp_num}\\{self.dataset_options.name}\\{self.model_name}\\sequence"
        self.save_path_raw = f"{self.save_path}\\raw"
        self.lastEpoch = 0
        # self.data_array=[]
        self.graphDataset()

    def graphDataset(self):
        dataset = dg.getDataset(self.dataset_options.name, self.dataset_options)
        plot_name = f"Original Dataset | Dataset: {self.dataset_options.name}"
        Vis.graphDataset(
            dataset=dataset,
            save_path=self.save_path_raw,
            plot_name = plot_name,
            dataset_options=self.dataset_options)

    def on_epoch_end(self, epoch, logs=None):
        epochStr = str(epoch).zfill(self.epoch_zFill)
        acc = logs["val_acc"]
        filename=f"ep-{epochStr}_vAcc-{acc:.2f}"
        plot_name = f"Epoch: {epochStr} | Accuracy: {(acc*100):.2f}% | Dataset: {self.dataset_options.name}"
        # data_arr = [self.model, filename, plot_name]
        # self.data_array.append(data_arr)
        self.graphModelPredictions(filename, plot_name)
        self.lastEpoch = epoch

    def on_train_end(self, logs=None):
        acc = logs["val_acc"]
        filename = f"final_eps-{self.lastEpoch+1}_vAcc-{acc:.2f}"
        plot_name = f"Final | Epochs: {self.lastEpoch+1} | Accuracy: {(acc*100):.2f}% | Dataset: {self.dataset_options.name}"
        self.graphModelPredictions(filename=filename, plot_name=plot_name)
        file_name = f"sequence_exp-{self.exp_num}_{self.model_name}.gif"
        save_path = f"{self.save_path}\\{file_name}"

        plot_list = os.listdir(self.save_path_raw)
        duration_arr = np.full(shape=len(plot_list), fill_value=0.5)
        duration_arr[[0, len(plot_list)-1]] = 3
        duration_list = list(duration_arr)

        Vis.createSequence(image_path=self.save_path_raw, save_path=save_path, duration_list=duration_list)

    def graphModelPredictions(self, filename, plot_name):
        Vis.graphPredictions(
            dataset=self.dataset,
            model=self.model,
            save_path=self.save_path_raw,
            name=filename,
            plot_name=plot_name,
            dataset_options=self.dataset_options)