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
import Utils.filesaver as fs

class LandscapeGenerator():

    def __init__(self, seed:int=1):
        """Initialize the landscape generator with the given seed
        Create the directory for saving all landscape files and the vis files

        Args:
            seed:int - the seed for the gen
        """
        self.seed = seed
        filesaver = fs.FileSaver(directory=".local/landscapes", base_name="landscape", zfill=4)
        self.directory = filesaver.getNextPath(createDir=True)

        self.vis_filesaver = fs.FileSaver(f"{self.directory}/visualizations", "vis")

    def generateLandscape(
        self,
        model_path:str,
        dMin:int,
        dMax:int,
        modelSideLength:int,
        dataset:np.ndarray=None,
        dataset_options:DataTypes=None
        ):
        """Generate the loss landscape for the provided model

        Args:
            model_path:str - the path to the model to generate the landscape for
            dMin:int - the minimum value for the direction vector scalars
            dMax:int - the maxmimum value for the direction vector scalars
            modelSideLength:int - the dimensions of the lxl square to use to generate the models (so the square side length)
            
            dataset:np.ndarray - optionally, the dataset to use for landscape generation
        """
        model = tf.keras.models.load_model(model_path)
        directions = dir.TrainableDirections(model, model_path, ds_options, seed=seed)
        directions.createDirections()
        model_gen = directions.alteredModelGenerator(dMin, dMax, modelSideLength)

        loss_grid = np.zeros((modelSideLength, modelSideLength))
        dataset = dataset if dataset is not None else dg.getDataset(options=dataset_options)

        self.fillLossGrid(loss_grid, model_gen, dataset)

        lin = np.linspace(dMin, dMax, modelSideLength)
        xvals, yvals = np.meshgrid(lin, lin)

        self.plotLossGrid(loss_grid, xvals, yvals)

    def plotLossGrid(self, loss_grid, xvals, yvals):
        # loss is between 0 and 1
        contourf = plt.contourf(X=xvals, Y=yvals, Z=loss_grid, vMin=0, vMax=1, cmap="YlOrRd")
        file_path = f"{self.directory}/landscape.png"
        plt.savefig(file_path, bbox_inches='tight', pad_inches=0.25, dpi=400)
        plt.show()

    def fillLossGrid(self, loss_grid, model_gen, dataset):

        hasNext = True
        data, labels = dataset
        counter = 0
        while hasNext:
            try:
                new_model, scalars, coords = next(model_gen)
                x, y = coords
                alpha, beta = scalars
                loss_metrics = new_model.evaluate(data, labels)
                loss = loss_metrics[0]
                print(loss)
                loss_grid[x, y] = loss

                # vis_save_path = f".local/landscape/visualizations/landscape-{exp_num}"
                vis_name = f"{counter}-({alpha}, {beta})"
                vis_save_path = self.vis_filesaver.getFilePath(vis_name)
                
                vis.graphPredictions(dataset, new_model, vis_save_path, vis_name, save_figure=True)
                print(counter)
                counter += 1
            except (StopIteration):
                hasNext = False
            if counter % 50 == 0:
                gc.collect() # garbage collector periodically because paranoia

    # def getWeights(model:tf.keras.models) -> list:
    #     weightList = []

    #     for index, layer in enumerate(model.layers):
    #         weightList.append(layer.weights)
    #         # np.append(weightArr, layer.get_weights())
    #     return weightList



model_path = ".local/models/exp-0/model-0002-[4, 4, 4, 4]/model"
dMin = -10
dMax = 10
dNumPoints = 300
sideLength = 50
ds_options = dg.PolynomialOptions(numPoints=dNumPoints)
seed=3
#exp_num = 8 # this needs to be incremented each time until i automate it

# dataset = dg.getDataset(options=ds_options)
# Polynomial.plotPolynomial(ds_options.coefficients, ds_options.dMin, ds_options.dMax, dataset)

# plt.scatter(points[:,0], points[:,1])
# plt.show()

# generateLandscape(model_path, wMin, wMax, sideLength, dataset_options=ds_options, seed=seed)