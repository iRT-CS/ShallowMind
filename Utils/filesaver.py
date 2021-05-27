import os
from pathlib import Path


class FileSaver():

    def __init__(self, directory:str, base_name:str, zfill:int=None):
        """
        Initializes the file saver with the given directory, zfill, and base name
        Also creates the given directory if it doesn't already exist

        The filesaver is primarily useful for directories where zfilled identifiers are used,
        but can also be used with manual identifiers

        Args:
            directory:str - the directory for the file saver
            zfill:int - the amount to zfill the ids by
            base_name:str - the base name for files in the folder
        """
        self.directory = directory
        self.zfill = zfill
        self.base_name = base_name
        Path(self.directory).mkdir(parents=True, exist_ok=True)
    
    def getLatestId(self) -> str:
        """
        Gets the latest file id from the directory. If there's nothing in the folder,
        None is returned.

        Returns:
            The string of the file id if there are files with the base name and ids present in the folder
            Otherwise, -1
        """
        
        fileList = self.getFileList()
        
        if len(fileList) == 0:
            # indexStr = str("0").zfill(self.zfill)
            return -1
        else:
            latestFolder = fileList[len(fileList)-1]
            start_index = len(self.base_name) + 1 # +1 for dash
            end_index = start_index+self.zfill
            index = int(latestFolder[start_index:end_index])
            indexStr = str(index).zfill(4)
        return indexStr
    
    def incrementId(self, id:str=None, inc=1):
        """
        Increments an id by a specified amount

        Args:
            id:str - the id to increment. If none is provided, then the latest is used

        """
        id = self.getLatestId() if id is None else id
        idInt = int(id)
        incrementedId = idInt + inc

        incStr = str(incrementedId).zfill(self.zfill)
        return incStr

    def getFileById(self, id:str):
        """
        Finds a file with a specified id from the folder

        Args:
            id:str - the id of the file to find
        
        Returns:
            str - the file with the specified id from the directory

        """
        fileList = self.getFileList()
        for file in fileList:
            if id in file:
                return file
        return None
        

    def getFileList(self) -> list:
        """
        Gets all the files in the folder with the base name in them

        Returns:
            A list of all the file names within the folder with the base name
        """
        fullFileList = os.listdir(self.directory)
        fullFileList.sort()
        fileList = []
        for file in fullFileList:
            if self.base_name in file:
                fileList.append(file)
        return fileList
    
    def getFilePath(self, identifier:str, additional:str=""):
        """
        Gets a full file path with the provided id and optional additional identifier

        Args:
            identifier:str - the id of the file, or the full identifier containing other info
            additional:str - the optional additional identifer of the file
        
        Return:
            str - the full file path with the base name, id, and identifier
        """
        return f"{self.directory}/{self.base_name}-{identifier}{additional}"
    
    def getNextPath(self, additional:str="", createDir:bool=False):
        """
        Convenience method to get the next full path with the next id

        Args:
            additional:str - optional addition to the filepath
            createDir:bool - whether or not to create the directory, defaults to false
        
        Returns:
            str - full filepath for the next id
        """

        nextId = self.incrementId(inc=1)
        filepath = self.getFilePath(nextId, additional)
        if createDir:
            Path(filepath).mkdir(parents=True, exist_ok=True)
        return filepath

# path=".local/filesaver_test"
# base_name = "file"
# zfill=4
# filesaver = FileSaver(directory=path, base_name=base_name, zfill=zfill)

# print(filesaver.getLatestId())
# print(filesaver.incrementId(inc=1))
# print(filesaver.getFileList())
# filepath = filesaver.getFilePath(filesaver.incrementId(inc=1))

# with open(f"{filepath}.txt", "w") as fp:
#     pass