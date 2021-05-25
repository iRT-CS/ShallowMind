import os
from pathlib import Path
import sys
import gc

path_to_file = str(Path(os.path.abspath(os.path.dirname(__file__))).parent.absolute())
sys.path.insert(0, path_to_file)

from Datasets.DatasetGenerator import DataTypes
import Datasets.DatasetGenerator as dg
import Landscape.TrainableDirections as dir
# import Landscape
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from Datasets import Polynomial
import Vis as vis
import matplotlib.colors as cmp

def generateLandscape(model_path:str, vMin, vMax, numPoints, dataset:np.ndarray=None, dataset_options:DataTypes=None, seed:int=1):
    model = tf.keras.models.load_model(model_path)
    directions = dir.TrainableDirections(model, model_path, ds_options, seed=seed)
    directions.createDirections()
    model_gen = directions.alteredModelGenerator(vMin, vMax, numPoints)

    loss_grid = np.zeros((numPoints, numPoints))
    dataset = dataset if dataset is not None else dg.getDataset(options=dataset_options)

    fillLossGrid(loss_grid, model_gen, dataset)

    lin = np.linspace(vMin, vMax, numPoints)
    xvals, yvals = np.meshgrid(lin, lin)

    plotLossGrid(loss_grid, xvals, yvals)

def plotLossGrid(loss_grid, xvals, yvals):
    contourf = plt.contourf(loss_grid, vmin=0, vmax=1, cmap="YlOrRd")
    plt.savefig(f".local/landscape/visualizations/landscape-{exp_num}/landscape.png", bbox_inches='tight', pad_inches=0.25, dpi=400)
    plt.show()

def fillLossGrid(loss_grid, model_gen, dataset):

    hasNext = True
    data, labels = dataset
    counter = 0
    while hasNext:
        try:
            new_model, coords = next(model_gen)
            x, y = coords
            loss_metrics = new_model.evaluate(data, labels)
            loss = loss_metrics[0]
            print(loss)
            loss_grid[x, y] = loss

            vis_save_path = f".local/landscape/visualizations/landscape-{exp_num}"
            vis_name = f"nm-{counter}-({x}, {y})"
            vis.graphPredictions(dataset, new_model, vis_save_path, vis_name, save_figure=True)
            print(counter)
            counter += 1
        except (StopIteration):
            hasNext = False
        if counter % 50 == 0:
            gc.collect()

# def getWeights(model:tf.keras.models) -> list:
#     weightList = []

#     for index, layer in enumerate(model.layers):
#         weightList.append(layer.weights)
#         # np.append(weightArr, layer.get_weights())
#     return weightList



model_path = ".local/models/exp-0/model-0002-[4, 4, 4, 4]/model"
wMin = -10
wMax = 10
dNumPoints = 300
sideLength = 50
ds_options = dg.PolynomialOptions(numPoints=dNumPoints)
seed=6
exp_num = 10 # this needs to be incremented each time until i automate it

# dataset = dg.getDataset(options=ds_options)
# Polynomial.plotPolynomial(ds_options.coefficients, ds_options.vMin, ds_options.vMax, dataset)

# plt.scatter(points[:,0], points[:,1])
# plt.show()

generateLandscape(model_path, wMin, wMax, sideLength, dataset_options=ds_options, seed=seed)