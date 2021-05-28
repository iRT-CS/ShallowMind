

from pathlib import Path
from typing import OrderedDict
import numpy as np


class VtkWriter():

    def __init__(self, pathToFile, filename):
        """Initializes the vtp writer.

        Args:
            pathToFile:str - the path to the file's parent directory, not including the filename and ext or anything
            filename:str - the file name, without the extension
        """
        self.filepath = f"{pathToFile}/{filename}.vtk"
        # create file if it doesnt exist
        Path(pathToFile).mkdir(parents=True, exist_ok=True)
        f = open(self.filepath, "w")
        f.close()
    
    def writeVtk(self, VtkFormat):
        with open(self.filepath, "w") as file:
            format_str = VtkFormat.convertDataToString()
            file.write(format_str)

    

class VtkDataFormats():
    
    STRUCTURED_GRID = "STRUCTURED_GRID"

    def __init__(self, description, dataset, dimensions):
        self.format_dict = {
            "ASCII": description,
            "DATASET": dataset,
            "DIMENSIONS": dimensions,
        }


class StructuredGrid(VtkDataFormats):

    name = VtkDataFormats.STRUCTURED_GRID

    def __init__(self, dimensions:str, numPoints:int, dataType, dataPoints:np.ndarray, description:str=""):
        super().__init__(description=description, dataset=self.name, dimensions=dimensions)
        self.format_dict["POINTS"] = f"{numPoints} {dataType}"
        self.format_dict["DATA_POINTS"] = dataPoints
    
    def convertDataToString(self):
        formatStr = ""
        for key, value in self.format_dict.items():
            if key != "DATA_POINTS":
                formatStr += f"{key} {value}\n"
            else:
                dataStr = self.getDataString(value)
                formatStr += f"{dataStr}"
    
    def getDataString(self, data_arr:np.ndarray):
        dataStr = ""
        for index, x in np.ndenumerate(data_arr):
            dataStr += f"{(' '.join(index))} {x}\n"
        
        return dataStr
