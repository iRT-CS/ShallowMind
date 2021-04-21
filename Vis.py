r"""
Setup venv:
    $ Set-ExecutionPolicy -Scope CurrentUser remotesigned
    $ .local\.venv\Scripts\Activate.ps1

launch tensorboard (w anaconda):
    - open anaconda powershell
    - enter: tensorboard --logdir='C:\Users\okt28\OneDrive\Compsci_Main\ShallowMind\Clone\ShallowMind\.local\logs'

common errors:

    Error: Could not find module '[ommitted...]\ShallowMind\.venv\Library\bin\geos_c.dll' (or one of its dependencies).
           Try using the full path with constructor syntax.
    Fix: DISABLE ANACONDA, its broken. Do "conda deactivate" in vsc terminal
"""

from tensorflow.python.keras.backend import switch
from numpy.random import seed as np_seed
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import keras
import datetime
import time
import Datasets.GaussianBoundary as gb
from Utils import seeding
import os
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import Utils.VisColormaps as vcmap
from Datasets import DatasetGenerator as dg
import imageio



def graphDataset(dataset:np.ndarray, save_path:str, plot_name:str=None, dataset_options=None, shouldSave:bool=True):
    """Graphs the dataset provided and saves

    Args:
        dataset: np.ndarray - the dataset to graph
        save_path: str - the save location
        plot_name: str - the name of the plot
        shouldSave: bool - whether the figure should be saved, otherwise directly plt.show()'d. Defaults to True
    """
    print(f"Dataset boundary, seed {seeding.getSeed()}")
    name = f"!ds_seed-{seeding.getSeed()}"
    coords, labels = dataset
    xcoords = coords[:,0]
    ycoords = coords[:,1]
    fig = plt.figure()
    ax = fig.add_subplot()
    scatter = ax.scatter(xcoords, ycoords, s=10, c=labels, cmap="RdYlBu")
    # plt.show()
    if plot_name is not None:
        plt.title(plot_name, loc="left")
    if dataset_options is not None:
        dg.setDatasetBoundaryPlot(ax, options=dataset_options)

    if shouldSave:
        saveFigure(save_path=save_path, figure=fig, name=name)
    else:
        plt.show()

    plt.close()

def graphPredictions(dataset:np.ndarray, model:tf.keras.models, save_path:str, name:str, plot_name:str=None, dataset_options=None):
    """Graphs the given model's predictions agaisnt the actual results
    also displays confidence in predictions as a contour map

    Args:
        dataset: np.ndarray - the dataset to check predictions for
        model: tf.keras.model - the model use
        save_path: str - the path to save the visualizations to
        name: str - the name of the model
    """
    print(f"Prediction boundary, b_{name}, seed {seeding.getSeed()}")
    save_name = f"b_{name}"
    fig = plt.figure()
    ax = fig.add_subplot()

    coords, y_true = dataset
    y_pred = model.predict(coords)
    y_pred_rounded = np.around(y_pred)

    # define bounds of the domain
    min1, max1 = coords[:, 0].min()-1, coords[:, 0].max()+1
    min2, max2 = coords[:, 1].min()-1, coords[:, 1].max()+1

    # define the x and y scale
    x1grid = np.arange(min1, max1, 0.1)
    x2grid = np.arange(min2, max2, 0.1)

    # create all of the lines and rows of the grid
    xx, yy = np.meshgrid(x1grid, x2grid)

    # flatten each grid to a vector
    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))

    # horizontal stack vectors to create x1,x2 input for the model
    grid = np.hstack((r1,r2))

    # make predictions for the grid
    y_grid = model.predict(grid)
    # reshape the predictions back into a grid
    zz = y_grid.reshape(xx.shape)

    # plot the grid of x, y and z values as a surface
    # issue with this was that it wasnt scaled
    # old RdBu
    cmap = vcmap.createContourColormap()
    plt.contourf(xx, yy, zz, cmap=cmap, vmin=0, vmax=1)
    redCmp1 = mpl.cm.get_cmap("seismic")
    redCmp2 = mpl.cm.get_cmap("YlOrRd")
    blueCmp =  mpl.cm.get_cmap("coolwarm")
    # old RdYlBu
    # dark points were classified as being that color, while theyre actually the other one
    # ^ i have no idea what that means
    # red right, blue right, red wrong, blue wrong
    # color = [redCmp2(0.65), blueCmp(0.20), redCmp1(0.93), blueCmp(0)]
    color = [redCmp2(0.65), blueCmp(0.15), redCmp1(0.93), blueCmp(0)]
    # create scatter plot for samples from each class
    for class_value in range(2):
        # get row indexes for samples with this class
        true_rows = np.where(y_true == class_value)
        # get rows where model predicted this class
        predicted_rows = np.where(y_pred_rounded == class_value)
        correct_rows = np.intersect1d(true_rows[0], predicted_rows[0])
        incorrect_rows = np.array(np.setdiff1d(predicted_rows[0], true_rows[0]))
        # create scatter of these samples
        # plt.scatter(coords[row_ix, 0], coords[row_ix, 1], s=10, color=color[class_value])
        # plot correct predictions
        # edgeColor = "#FFFFFF"
        # lineWidth = 0.04
        size = 12
        plt.scatter(coords[
            correct_rows, 0], coords[correct_rows, 1], s=size, color=color[class_value])
        # plot incorrect predictions
        plt.scatter(
            coords[incorrect_rows, 0], coords[incorrect_rows, 1], s=size,
            color=color[class_value+2])
    # plt.show()
    if plot_name is not None:
        plt.title(plot_name, loc="left")
    if dataset_options is not None:
        dg.setDatasetBoundaryPlot(ax, options=dataset_options)

    # plt.show()
    saveFigure(save_path=save_path, name=save_name, figure=fig)
    plt.close()

"""Saves a figure to the given path with the given filename
:param save_path: str - the path to save the file to
:param figure: matplotlib.pyply.Figure - the figure to save
:param name: the filename to save the figure as
"""
def saveFigure(save_path:str, figure:plt.Figure, name:str):
    Path(save_path).mkdir(parents=True, exist_ok=True)
    file_path = f"{save_path}\\{name}.png"
    figure.savefig(file_path, bbox_inches='tight', pad_inches=0.25, dpi=400)
    

"""Gets visualizations for a model
:param model: tf.keras.model - the model to use
:param save_path: str - the path to save the visualizations to
:param name: str - the name of the model, used for saving and organization
:param dataset_options: dg.DataTypes(.options) - the options for dataset generation
:param plotDataset: boolean - whether to plot the generated dataset or not
"""
def getVisualizations(model:tf.keras.models, save_path:str, name:str, dataset_options:dg.DataTypes, plotDataset:bool =False):
    dataset = dg.getDataset(dataset_options.name, dataset_options)
    if plotDataset:
        index = save_path.index("dataset-")
        data_save_path = save_path[:save_path.index("\\", index)]
        graphDataset(dataset, data_save_path, dataset_options=dataset_options)
    graphPredictions(dataset=dataset, model=model, save_path=save_path, name=name, dataset_options=dataset_options)


def createSequence(image_path, save_path, duration_list):
    writer = imageio.get_writer(save_path, mode='I', duration=duration_list)
    plot_list = os.listdir(image_path)
    for plot in plot_list:
        # use long for the first and last
        image = imageio.imread(f"{image_path}\\{plot}")
        writer.append_data(image)