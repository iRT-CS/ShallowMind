
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

    def __init__(self, trained_model:tf.keras.models, model_path:str, ds_options:dg.DataTypes=None, dataset:np.ndarray=None, seed=1):
        self.trained_model = trained_model
        self.md_trainable = trained_model.trainable_variables
        self.model_path = model_path
        self.ds_options = ds_options
        self.seed = seed
    
    def createDirections(self):
        self.dir1 = self.getDirectionalVector(self.seed)
        self.dir2 = self.getDirectionalVector(self.seed + 1)
    
    def getDirectionalVector(self, seed):
        np.random.seed(seed)
        tf.random.set_seed(seed)
        random_model = tf.keras.models.clone_model(self.trained_model)
        dataset = dg.getDataset(options=self.ds_options)
        vis.graphPredictions(dataset, random_model, "Landscape/vis", f"random_model-{seed}")
        rd_trainable = random_model.trainable_variables

        dir_vectors = []

        for rd_vect, md_vect in zip(rd_trainable, self.md_trainable):
            unnormalized_vect = rd_vect - md_vect # maybe other way around
            diff_vect = unnormalized_vect / tf.math.abs(unnormalized_vect)
            dir_vectors.append(diff_vect)

        return dir_vectors
    
    def gridScalarGenerator(self, dMin, dMax, numValues):
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
            

                


