# Script to read dill data objects as numpy arrays.
from PIL import Image
import os
import sys
import logging
from random import randint

root = logging.getLogger()
root.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

from mapdataset import ImageGroupReader, single_layer_converter, MapsDataset, MapReader



# dillFolder = "./data/output/SF_Layered/32x32/group-1280-stride-10"
# nGroups = 0
# # Iterate directory
# for path in os.listdir(dillFolder):
#     # check if current path is a file
#     if os.path.isfile(os.path.join(dillFolder, path)) and path.endswith(".dill"):
#         nGroups += 1

# for i in range(nGroups):
#     reader = ImageGroupReader(dillFolder)
#     data = reader.load_group(groupNo=i)
#     patchImgArray = reader.asImg(data[0])
#     im = Image.fromarray(patchImgArray)
#     path = os.path.join(dillFolder, f"{i}-0.png")
#     im.save(path)

mapsDataset = MapsDataset(
    patch_size=(32, 32), 
    stride=10, 
    sample_group_size=1280, 
    converter=single_layer_converter,
    outputDir="./data/output"
    ) 

mapsDataset.loadPatches("./data/output/SF_Layered/32x32/group-1280-stride-10")
patchNo = randint(0, len(mapsDataset))
logging.info(f"reading patch {patchNo}")
patch = mapsDataset[patchNo]
    