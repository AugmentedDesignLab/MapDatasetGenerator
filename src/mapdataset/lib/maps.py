from torch.utils.data import Dataset
from PIL import Image
import torch.nn.functional as F
import os
from tqdm.notebook import tqdm
import torch
import numpy as np
import random
import dill
import logging


# Credit - Bahar
# Modifications - Muktadir
class MapReader:
    def __init__(self, filename, mapName):
        self.mapName = mapName
        f = open(filename, 'r')
        data = f.read().split()
        self.size = (int(data[0]), int(data[1]))
        self.data = [[int(data[i * self.size[1] + j + 2]) for j in range(self.size[1])] for i in range(self.size[0])]
        logging.info(f"Your map is {self.size[0]}x{self.size[1]}")
    

    def standardize(self, converter):
        logging.info(f"Normalizing the data in [-1, 1] using {converter.__class__.__name__}")
        #TODO Ishaan
        #  (self.converter.get_char(mapReader.data[i + x][j + y]) / (len(self.converter.char_groups) - 1)) * -2 + 1
        maxVal = len(converter.char_groups) - 1
        for i in range(0, self.size[0]):
            for j in range(0, self.size[1]):
                self.data[i][j] = (converter.get_char(self.data[i][j]) / maxVal) * -2 + 1
        pass

# Converter object that can return character for OpenStreetMap layer group
# Credit - Bahar
class LayerToCharConverter:
    def __init__(self, char_groups=[[0], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]):
        self.char_groups = char_groups
        self.char_size = len(self.char_groups)

    def get_char(self, layer):
        for i in range(self.char_size):
            if layer in self.char_groups[i]:
                return i
        raise Exception("layer not available in char groups: {}".format(layer))

multi_layer_converter = LayerToCharConverter([[0], [1, 2, 3, 4, 5], [6, 7], [8, 9, 10, 11, 12]])
single_layer_converter = LayerToCharConverter([[0], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]])

# New Cell
# Credit - Bahar
# Modifications - Ishaan, Muktadir

class MapsDataset(Dataset):
    def __init__(self, patch_size, stride, sample_group_size, converter, outputDir="../data/output"):
        self.outputDir = outputDir
        self.char_size = converter.char_size
        self.converter = converter
        self.patch_size = patch_size
        self.stride = stride
        self.sample_group_size = sample_group_size
        self.samples = []
        self.block_size = self.patch_size[0] * self.patch_size[1] - 1

        os.makedirs(self.outputDir, exist_ok=True)

        


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = (self.samples[idx], self.samples[idx], self.samples[idx])
        return torch.from_numpy(sample).unsqueeze(0)
    
    def add(self, mapReader):
        mapReader.standardize(self.converter)
        for i in range(0, mapReader.size[0] - self.patch_size[0] + 1, self.stride):
            for j in range(0, mapReader.size[1] - self.patch_size[1] + 1, self.stride):
                self.samples.append(self.extractSample(mapReader, topLeft=(i, j)))

    def loadPrecomputed(self, patchDirectory):
        #TODO
        pass
    
    #Generate image patches and write to data/output directory
    def generate_patches(self, mapReader, image_groups=3, outDirectory=None):
        """_summary_

        Args:
            mapReader (MapReader): reader for a single big map!
            image_groups (int, optional): _description_. Defaults to 3.
        """

        mapReader.standardize(converter=self.converter)

        if outDirectory is None:
            outDirectory = os.path.join(self.outputDir, mapReader.mapName, f"{self.patch_size[0]}x{self.patch_size[1]}", f"stride-{self.stride}")
        os.makedirs(outDirectory, exist_ok=True)

        img_group_number = 0
        for i in range(0, mapReader.size[0] - self.patch_size[0] + 1, self.stride):
            for j in range(0, mapReader.size[1] - self.patch_size[1] + 1, self.stride):

                sample = self.extractSample(mapReader, topLeft=(i, j))

                self.samples.append(sample)

                if len(self.samples) == self.sample_group_size:
                    path = os.path.join(outDirectory, str(img_group_number) + ".dill")
                    with open(path, 'wb+') as f:
                        dill.dump(self.samples, f)
                        f.close()    
                    logging.info(f"Image group {img_group_number} saved in {path}")
                    self.samples.clear()
                    img_group_number+=1
        pass

    def extractSample(self, mapReader, topLeft):
        i = topLeft[0]
        j = topLeft[1]
        # sample = [
        #             [
        #                     (self.converter.get_char(mapReader.data[i + x][j + y]) / (len(self.converter.char_groups) - 1)) * -2 + 1 # TODO this conversion should be done once in the original data instead of patches.
        #                 for y in range(self.patch_size[1])
        #             ]
        #             for x in range(self.patch_size[0])
        #         ]
        sample = [
                    [
                            mapReader.data[i + x][j + y]
                        for y in range(self.patch_size[1])
                    ]
                    for x in range(self.patch_size[0])
                ]
        return np.asarray(sample)


    def shuffle(self):
        random.shuffle(self.samples)

    def get_train_test(self, train_ratio):
        train_dataset = MapsDataset(self.patch_size, self.stride, self.converter)
        test_dataset = MapsDataset(self.patch_size, self.stride, self.converter)
        train_size = int(len(self.samples) * train_ratio)
        train_dataset.samples = self.samples[:train_size]
        test_dataset.samples = self.samples[train_size:]
        return train_dataset, test_dataset

# Some utility functions
def img_to_tensor(im):
  return torch.tensor(np.array(im.convert('RGB'))/255).permute(2, 0, 1).unsqueeze(0) * 2 - 1

def tensor_to_image(t):
  return Image.fromarray(np.array(((t.squeeze().permute(1, 2, 0)+1)/2).clip(0, 1)*255).astype(np.uint8))

def gather(consts: torch.Tensor, t: torch.Tensor):
    """Gather consts for $t$ and reshape to feature map shape"""
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1, 1)

def map_img_to_tensor(im):
  return torch.tensor(np.array(im.convert('L'))/255).unsqueeze(0) * 2 - 1

def map_tensor_to_image(t):
  return Image.fromarray(np.array(((t.squeeze()+1)/2).clip(0, 1)*255).astype(np.uint8))