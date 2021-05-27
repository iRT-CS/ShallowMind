
import os
from pathlib import Path
import sys

path_to_file = str(Path(os.path.abspath(os.path.dirname(__file__))).parent.absolute())
sys.path.insert(0, path_to_file)

import numpy as np
import tensorflow as tf
from Datasets.DatasetGenerator import DataTypes
import Datasets.DatasetGenerator as dg
import Vis as vis

class TrainableDirections():

    def __init__(
        self,
        trained_model:tf.keras.models,
        model_path:str,
        ds_options:dg.DataTypes=None,
        dataset:np.ndarray=None,
        seed=1):
        """Initializes the directions

        Args:
            trained_model:tf.keras.models - the trained model to use
            model_path:str - the string to the trained model
            ds_options:dg.DataTypes - the dataset options to use
            dataset:np.ndarray - the dataset to use, either options or dataset must be provided
            seed:int - the seed

        """

        self.trained_model = trained_model
        self.md_trainable = trained_model.trainable_variables
        self.model_path = model_path
        self.ds_options = ds_options
        self.seed = seed
    
    def createDirections(self, rand_vis_directory:str):
        """Creates two random direcions and stores them as attributes
        Uses the instance seed variable for the first direction and seed+1 for the second

        Args:
            rand_vis_directory:str - the directory to save the predictions from the initial random models to
        """
        self.dir1 = self.getDirectionalVector(self.seed, rand_vis_directory)
        self.dir2 = self.getDirectionalVector(self.seed + 1, rand_vis_directory)
    
    def getDirectionalVector(self, seed, rand_vis_directory):
        """Creates a directional vector for the model
        The direction is filter normalized, so its magnitude is the same as the magnitude of
        the corrosponding vecor from the original model

        Args:
            seed:int - the seed to use
            rand_vis_directory:str - the directory to save random model visualizations to

        Returns:
            list - a list of the tensors comprising the directional vector
        """
        np.random.seed(seed)
        tf.random.set_seed(seed)
        random_model = tf.keras.models.clone_model(self.trained_model)
        dataset = dg.getDataset(options=self.ds_options)
        vis.graphPredictions(dataset, random_model, f"{rand_vis_directory}", f"random_model-{seed}")
        rd_trainable = random_model.trainable_variables

        dir_vectors = []

        for rd_vect, md_vect in zip(rd_trainable, self.md_trainable):
            unnormalized_vect = rd_vect - md_vect # maybe other way around
            diff_vect = (unnormalized_vect / tf.math.abs(unnormalized_vect)) * tf.math.abs(md_vect) # filter normalized
            dir_vectors.append(diff_vect)

        return dir_vectors
    
    def gridScalarGenerator(self, dMin:int, dMax:int, numValues:int):
        """A generator to provide alpha and beta values between dMin and dMax to
        parameterize the directions

        Args:
            dMin:int - the minimum value for alpha and beta
            dMax:int - the maximum value for alpha and beta
            numValues:int - the number of evenly spaced values to generate between dMin and dMax
        
        Yields:
            two tuples, one of alpha and beta and the other of their corrosponding coordiantes on the loss grid
        """
        lin = np.linspace(dMin, dMax, numValues)
        xvals, yvals = np.meshgrid(lin, lin)
        for i in range(numValues):
            for j in range(numValues):
                alpha = xvals[i, j]
                beta = yvals[i, j]
                print(f"{alpha}, {beta}")
                print(f"({i}, {j})")
                yield (alpha, beta), (i, j)
    
    def alteredModelGenerator(self, dMin, dMax, numValues):
        """Generator for models altered by the directional tensors

        Args:
            dMin:int - the minimum value of alpha and beta
            dMax:int - the maximum value of alpha and beta
            numValues:int - the number of values between alpha and beta
        
        Yields:
            tf.keras.models, tuple, tuple - the model altered by the directional vectors,
                                            a tuple of the scalars used, and the points on the loss grid
        """
        grid_gen = self.gridScalarGenerator(dMin, dMax, numValues)
        model_copy = tf.keras.models.load_model(self.model_path)
        endLoop = False
        while not endLoop:
            try:
                scalars, coords = next(grid_gen)
                alpha, beta = scalars
                for mc_tensor, md_tensor, d1_tensor, d2_tensor in zip(model_copy.trainable_variables, self.md_trainable, self.dir1, self.dir2):
                    mc_tensor.assign(tf.add(md_tensor, tf.add(tf.multiply(d1_tensor, alpha), tf.multiply(d2_tensor, beta))))
                yield model_copy, scalars, coords
            except (StopIteration):
                endLoop = True
            

                
# lin = np.linspace(-1,1,2)
# xvals, yvals = np.meshgrid(lin, lin)
# _=0

