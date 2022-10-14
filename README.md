# MapDatasetGenerator
Generate and load dataset of road network maps.

<!-- # Quick start
* Use pip to install necessary python packages `pip install -r requirements.txt`.
* Run `python run.py` script. It will store .dill files in the data/output/ directory. 
* These dill files can be read using the `ImgGroupReader` object in `read.py`. Run `read.py` to run a small test for the same.  
 -->

# Installation from pip

```
pip install mapdatasetgenerator
```

# Creating patches
```
# Run this script to generate data in /output directory.
import logging
import sys

root = logging.getLogger()
root.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

from mapdataset import ImageGroupReader, single_layer_converter, MapsDataset, MapReader


sfMap = MapReader('./data/input/sf_layered.txt', "SF_Layered")
mapsDataset = MapsDataset(
    patch_size=(32, 32), 
    stride=10, 
    sample_group_size=1280, 
    converter=single_layer_converter,
    outputDir="./data/output"
    ) 
    
mapsDataset.generate_patches(sfMap) #This will generate dill files which contain the saved sample lists.

```

# Reading patches

```
# Script to read dill data objects as numpy arrays.
from PIL import Image
import os
import sys
import logging

root = logging.getLogger()
root.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

from mapdataset import ImageGroupReader, single_layer_converter, MapsDataset, MapReader



dillFolder = "./data/output/SF_Layered/32x32/stride-10"
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
```