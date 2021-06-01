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
import Utils.infologger as lg
from datetime import datetime
import Utils.vtkwriter as vtk

class LandscapeGenerator():

    def __init__(self, exp_num, seed:int=1):
        """Initialize the landscape generator with the given seed
        Create the directory for saving all landscape files and the vis files

        Args:
            seed:int - the seed for the gen
        """
        self.seed = seed
        self.exp_num = exp_num
        filesaver = fs.FileSaver(directory=f".local/landscapes/exp-{exp_num}", base_name="landscape", zfill=4)
        self.directory = filesaver.getNextPath(createDir=True)
        self.ls_id = filesaver.incrementId(inc=0)

        self.vis_filesaver = fs.FileSaver(f"{self.directory}/visualizations", "vis")
        self.logger = lg.InfoLogger(self.directory, "ls_log")

        self.vtkwriter = vtk.VtkWriter(self.directory, "ls_data")

    def generateLandscape(
        self,
        model_path:str,
        dMin:int,
        dMax:int,
        modelSideLength:int,
        dataset_options:DataTypes,
        save_vis:bool=True
        ):
        """Generate the loss landscape for the provided model

        Args:
            model_path:str - the path to the model to generate the landscape for
            dMin:int - the minimum value for the direction vector scalars
            dMax:int - the maxmimum value for the direction vector scalars
            modelSideLength:int - the dimensions of the lxl square to use to generate the models (so the square side length)
            dataset_options:dg.DataTypes - the options for the dataset to use for landscape generation
            save_vis:bool - whether or not to save visualizations
        """

        log_dict = {
            "computer": os.environ['COMPUTERNAME'],
            "model_path" : model_path,
            "seed": self.seed,
            "landscape_save_path": self.directory,
            "dMin" : dMin,
            "dMax": dMax,
            "sideLength": sideLength,
            "dataset_options": dataset_options.getInfoDict(),
            "time_begin": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
            "completed_gracefully": False,
            "save_visualizations": save_vis,
        }

        self.logger.writeInfo(log_dict)

        model = tf.keras.models.load_model(model_path)
        directions = dir.TrainableDirections(model, model_path, ds_options, seed=seed)
        directions.createDirections(f"{self.directory}/rand_vis")
        model_gen = directions.alteredModelGenerator(dMin, dMax, modelSideLength)

        loss_grid = np.zeros((modelSideLength, modelSideLength))
        dataset = dg.getDataset(options=dataset_options)

        self.fillLossGrid(loss_grid, model_gen, dataset, save_vis)

        lin = np.linspace(dMin, dMax, modelSideLength)
        xvals, yvals = np.meshgrid(lin, lin)
        self.saveVtk(loss_grid)
        self.saveNumpy(loss_grid, xvals, yvals)
        self.plotLossGrid(loss_grid, xvals, yvals)

    def saveVtk(self, loss_grid):
        vtkFormat = vtk.StructuredGrid(dataPoints=loss_grid, description=f"landscape-{self.exp_num}-{self.ls_id}")
        self.vtkwriter.writeVtk(vtkFormat)
    
    def saveNumpy(self, loss_grid, xvals, yvals):
        np.savetxt(f"{self.directory}/loss_grid.txt", loss_grid)
        np.savetxt(f"{self.directory}/xvals.txt", xvals)
        np.savetxt(f"{self.directory}/yvals.txt", yvals)

    def plotLossGrid(self, loss_grid, xvals, yvals):
        """Plot the loss contour grid from the values
        Saves two plots, one with a grid over it with the locations of the models evaluated and
        one without that grid.
        Also logs infomation about the completion

        Args:
            loss_grid:np.ndarray - the array of loss values to plot
            xvals:np.ndarray - the x values, or the alpha values, used in the landscape
            yvals:np.ndarray - the y values, or the beta values, used in the landscape
        """
        # loss is between 0 and 1
        contourf = plt.contourf(xvals, yvals, loss_grid, vMin=0, vMax=1, cmap="YlOrRd")
        pure_filepath = f"{self.directory}/landscape-pure.png"
        plt.savefig(pure_filepath, bbox_inches='tight', pad_inches=0.25, dpi=400)

        grid_filepath = f"{self.directory}/landscape-grid.png"
        point_size = .3
        plt.scatter(xvals, yvals, point_size)
        plt.savefig(grid_filepath, bbox_inches='tight', pad_inches=0.25, dpi=400)
        
        log_dict = {
            "completed_gracefully": True,
            "time_end": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
        }
        self.logger.writeInfo(log_dict)

        plt.show()

    def fillLossGrid(self, loss_grid, model_gen, dataset, save_vis) -> None:
        """Evaluates all the weights to fill the loss grid with values
        Changes the loss grid in place, so returns nothing

        Args:
            loss_grid:np.ndarray - empty grid with the shape of the loss grid
            model_gen:TrainableDirections.alteredModelGenerator - the generator used to get new models
            dataset:np.ndarray - the dataset to evaluate the models on
            save_vis:bool - whether or not to save visualizations
        """

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

                
                if save_vis:
                    # vis_save_path = f".local/landscape/visualizations/landscape-{exp_num}"
                    vis_name = f"{counter}-({round(alpha, 3)}, {round(beta, 3)})"
                    vis_save_path = self.vis_filesaver.directory
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
dMin = -20
dMax = 20
dNumPoints = 400
sideLength = 201 # must be (multiple of 20 [or maybe its 2]) + 1 in order to get 0,0 which we need
ds_options = dg.PolynomialOptions(numPoints=dNumPoints)
seed=3
exp_num = 1
#exp_num = 8 # this needs to be incremented each time until i automate it

# dataset = dg.getDataset(options=ds_options)
# Polynomial.plotPolynomial(ds_options.coefficients, ds_options.dMin, ds_options.dMax, dataset)

# plt.scatter(points[:,0], points[:,1])
# plt.show()
ls = LandscapeGenerator(exp_num, seed)
ls.generateLandscape(model_path, dMin, dMax, sideLength, dataset_options=ds_options, save_vis=False)

# plt.scatter([1, 2, 3], [1, 2, 3], .3)
# plt.show()