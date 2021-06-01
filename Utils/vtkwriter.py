

from ast import Str
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
            "HEADER_NO-TITLE": "# vtk DataFile Version 2.0",
            "DESCRIPTION_NO-TITLE": description,
            "TYPE_NO-TITLE": "ASCII",
            "DATASET": dataset,
            "DIMENSIONS": dimensions,
        }


class StructuredGrid(VtkDataFormats):

    name = VtkDataFormats.STRUCTURED_GRID

    def __init__(self, dataPoints:np.ndarray, description:str="", dataType="float"):

        numPoints = dataPoints.size
        dimensions = (' '.join(map(str, dataPoints.shape))) + " 1"

        super().__init__(description=description, dataset=self.name, dimensions=dimensions)
        self.format_dict["POINTS"] = f"{numPoints} {dataType}"
        self.format_dict["DATA_POINTS_NO-TITLE"] = self.getDataString(dataPoints)
    
    def convertDataToString(self):
        formatStr = ""
        for key, value in self.format_dict.items():
            if "NO-TITLE" not in key:
                formatStr += f"{key} {value}\n"
            else:
                formatStr += f"{value}\n"
        return formatStr
    
    def getDataString(self, data_arr:np.ndarray):
        dataStr = ""
        for index, x in np.ndenumerate(data_arr):
            dataStr += f"{(' '.join(map(str, index)))} {x}\n"
        
        return dataStr

# test_arr = np.array([[1, 2, 3], [3, 2, 1], [3, 3, 3]])

# vtk_format = StructuredGrid(test_arr, "title here")

# vtk_writer = VtkWriter(".local/vtk-test", "vtk-test")
# vtk_writer.writeVtk(vtk_format)