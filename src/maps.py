from torch.utils.data import Dataset
from PIL import Image
import torch.nn.functional as F
import os
from tqdm.notebook import tqdm
import torch
import numpy as np
import random
import dill


# Credit - Bahar
class MapReader:
    def __init__(self, filename):
        f = open(filename, 'r')
        data = f.read().split()
        self.size = (int(data[0]), int(data[1]))
        self.data = [[int(data[i * self.size[1] + j + 2]) for j in range(self.size[1])] for i in range(self.size[0])]

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
# Modifications - Ishaan

class MapsDataset(Dataset):
    def __init__(self, window_size, step_size, sample_group_size, converter):
        self.char_size = converter.char_size
        self.converter = converter
        self.window_size = window_size
        self.step_size = step_size
        self.sample_group_size = sample_group_size
        self.samples = []
        self.block_size = self.window_size[0] * self.window_size[1] - 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = (self.samples[idx], self.samples[idx], self.samples[idx])
        return torch.from_numpy(np.array(sample)).unsqueeze(0)
        #flat = torch.from_numpy(np.array(sample)).view(-1)
        #flat = flat[self.perm].float()
        #return flat
    
    def add(self, mapReader):
        for i in range(0, mapReader.size[0] - self.window_size[0] + 1, self.step_size):
            for j in range(0, mapReader.size[1] - self.window_size[1] + 1, self.step_size):
                self.samples.append([[
                    (self.converter.get_char(mapReader.data[i + x][j + y]) / (len(self.converter.char_groups) - 1)) * -2 + 1
                    for y in range(self.window_size[1])]
                    for x in range(self.window_size[0])])
    
    #Generate image patches and write to data/output directory
    def generate_patches(self, mapReader, image_groups=3):
        img_group_number = 0
        for i in range(0, mapReader.size[0] - self.window_size[0] + 1, self.step_size):
            for j in range(0, mapReader.size[1] - self.window_size[1] + 1, self.step_size):
                self.samples.append([[
                (self.converter.get_char(mapReader.data[i + x][j + y]) / (len(self.converter.char_groups) - 1)) * -2 + 1
                for y in range(self.window_size[1])]
                for x in range(self.window_size[0])])

                if len(self.samples)==self.sample_group_size:
                    with open("data/output/SF_group"+str(img_group_number)+".dill", 'wb+') as f:
                        dill.dump(self.samples, f)
                        f.close()    
                    print("Image group {} saved in data/output/ !".format(img_group_number))
                    self.samples.clear()
                    img_group_number+=1
        

    def shuffle(self):
        random.shuffle(self.samples)

    def get_train_test(self, train_ratio):
        train_dataset = MapsDataset(self.window_size, self.step_size, self.converter)
        test_dataset = MapsDataset(self.window_size, self.step_size, self.converter)
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