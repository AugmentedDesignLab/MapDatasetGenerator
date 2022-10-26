# Script to read dill data objects as numpy arrays.
import dill
import numpy as np
from PIL import Image
import os

from mapdataset import ImageGroupReader


dillFolder = "./data/output/SF_Layered/32x32/group-1280-stride-10"
nGroups = 0
# Iterate directory
for path in os.listdir(dillFolder):
    # check if current path is a file
    if os.path.isfile(os.path.join(dillFolder, path)) and path.endswith(".dill"):
        nGroups += 1

for i in range(nGroups):
    reader = ImageGroupReader(dillFolder)
    data = reader.load_group(groupNo=i)
    patchImgArray = reader.asImg(data[0])
    im = Image.fromarray(patchImgArray)
    path = os.path.join(dillFolder, f"{i}-0.png")
    im.save(path)
