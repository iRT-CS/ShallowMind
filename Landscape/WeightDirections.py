import os
from pathlib import Path
import sys
path_to_file = str(Path(os.path.abspath(os.path.dirname(__file__))).parent.absolute())
sys.path.insert(0, path_to_file)

import tensorflow as tf
import numpy as np

class DirectionalVectors():

    def __init__(self, modelWeights):
        self.modelWeights = modelWeights

    def generateRandomDirections(self):
        self.dir1 = self.getDirectionVector(self.modelWeights)
        self.dir2 = self.getDirectionVector(self.modelWeights)
        # direction2 = getDirectionVector(weightList)


    def getDirectionVector(self, weightList):
        vectorList = []
        initializer = tf.initializers.random_normal()

        for weight in weightList:
            rd_tensor_list = []
            for weightTensor in weight:
                rdWeight = initializer(shape=weightTensor.get_shape())
                # maybe other way around idk
                rdVect = weightTensor - rdWeight
                rd_tensor_list.append(rdVect)
            
            vectorList.append(rd_tensor_list)
        
        return vectorList
    
    def generateWeights(self, vMin, vMax, numValues):
        lin = np.linspace(vMin, vMax, numValues)
        xvals, yvals = np.meshgrid(lin, lin)
        for i in range(numValues):
            for j in range(numValues):
                alpha = xvals[i, j]
                beta = yvals[i, j]
                print(f"{alpha}, {beta}")

                alteredWeights = []
                for weightList, dir1_list, dir2_list in zip(self.modelWeights, self.dir1, self.dir2):
                    aw_tensor_list = []
                    for weightTensor, d1_tensor, d2_tensor in zip(weightList, dir1_list, dir2_list):
                        alteredTensor = weightTensor + d1_tensor * alpha + d2_tensor * beta
                        aw_tensor_list.append(alteredTensor)
                alteredWeights.append(aw_tensor_list)
                yield alteredWeights

# dv = DirectionalVectors([0])
# dv.generateWeights(0, 1, 3)