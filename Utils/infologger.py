import yaml
import collections
from pathlib import Path


class InfoLogger():
    """
    The info logger is used for storage of information about experiments and for easy programatic retrieval of
    information. Data ia stored in .yaml files.
    """
    def __init__(self, pathToFile, filename):
        """Initializes the info logger.

        Args:
            pathToFile:str - the path to the file's parent directory, not including the filename and ext or anything
            filename:str - the file name, without the extension
        """
        self.filepath = f"{pathToFile}/{filename}.yaml"
        # create file if it doesnt exist
        Path(pathToFile).mkdir(parents=True, exist_ok=True)
        f = open(self.filepath, "a+")
        f.close()

    def writeInfo(self, data_dict:dict, overwrite:bool=False):
        """Writes information from a dictionary into the yaml file
        Merges the old and new data from the file, unless specified otherwise

        Args:
            data_dict:dict - the dictionary of new data to write to the file
            overwrite:bool - if true, completely overwrites the file and only adds the new data. Defaults to false
        """

        if not overwrite:
            # get currrent data from file
            with open(self.filepath, "r") as file:
                current_data = yaml.full_load(file)
                current_data = {} if current_data is None else current_data
            
            # merge the old and new data
            current_data.update(data_dict)
            data_dict = current_data

        # update the file with the new data
        with open(self.filepath, "w") as file:
            yaml.dump(data_dict, file)
    
    def getInfo(self, key=None):
        """Gets information from the file
        If a key is provided, try to get data for the specified key
        Otherwise, return the whole data dictionary

        Args:
            key - optional, the key to get from the file. If it doesn't exist, None is returned
                  if not provided, the whole dictionary is returned
        
        Returns:
            dict | list | None - The information from the file, None if there was no data to get
        """
        with open(self.filepath, "r") as file:
            data_dict = yaml.full_load(file)
            if key is not None:
                return data_dict.get(key)
            return data_dict

# path = ".local/logger_test"
# filename = "polylog"
# logger = InfoLogger(path, filename)


# data_dict = {
#     "value": "val3",
#     "another value": "val2"
# }

# logger.writeInfo(data_dict)
# data = logger.getInfo("value2")
# print(data)
