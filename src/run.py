# Run this script to generate data in /output directory.
import maps

sfMap = maps.MapReader('../data/input/sf_layered.txt', "SF_Layerd")
mapsDataset = maps.MapsDataset((32, 32), 2, 1280, maps.single_layer_converter) #Third parameter is the group size
mapsDataset.generate_patches(sfMap) #This will generate dill files which contain the saved sample lists.
