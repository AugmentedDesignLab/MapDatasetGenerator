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

from lib import maps

sfMap = maps.MapReader('../data/input/sf_layered.txt', "SF_Layered")
mapsDataset = maps.MapsDataset((32, 32), 10, 1280, maps.single_layer_converter) #Third parameter is the group size
mapsDataset.generate_patches(sfMap) #This will generate dill files which contain the saved sample lists.
