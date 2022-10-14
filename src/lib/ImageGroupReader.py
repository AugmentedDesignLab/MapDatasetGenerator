import logging
import dill
import numpy as np
import os
import logging

class ImageGroupReader:
    def __init__(self, directory_path, location='SF'):
        self.data = []
        self.dir_path = directory_path
        self.location = location

    def getGroupPath(self, groupNo):
        return os.path.join(self.dir_path, f"{groupNo}.dill")

    def load_group(self, groupNo=0):
        with open(self.getGroupPath(groupNo), 'rb') as d:
            return dill.load(d)

           
    def groupAsImg(self, group):
        """Assumes the raw data is standardized. So, we project it back into PIL RGB image format

        Returns:
            _type_: _description_
        """
        # print(data[0])
        data = np.asarray(group)
        data = np.clip((data + 1) / 2, 0, 1) * 255
        data = data.astype(np.uint8)
        return data
    
    def asImg(self, patch):
        data = np.clip((patch + 1) / 2, 0, 1) * 255
        data = data.astype(np.uint8)
        return data