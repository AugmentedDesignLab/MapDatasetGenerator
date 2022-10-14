# Script to read dill data objects as numpy arrays.
import dill
import numpy as np
from PIL import Image
import torch


class ImgGroupReader:
    def __init__(self, directory_path, location='SF'):
        self.data = []
        self.dir_path = directory_path
        self.location = location

    def load_group(self, group=0):
        if self.dir_path[-1]=='/':
            with open(self.dir_path+str(group)+".dill", 'rb') as d:
                self.data = dill.load(d)
                d.close()
        else:
            self.dir_path = self.dir_path + "/"
            with open(self.dir_path+str(group)+".dill", 'rb') as d:
                self.data = dill.load(d)
                d.close()
           
    def read_numpy(self):
        if(self.data == []): return print("Image group file not loaded! Call load_image_group() first.")
        else:
            self.data = np.array(self.data)
            self.data = np.clip((self.data+1)/2, 0, 1)*255
            self.data = self.data.astype(np.uint8)
        return self.data

sf_groups = 117
for i in range(sf_groups):
    reader = ImgGroupReader("../data/output/city_patches/dill/washington")
    reader.load_group(group=i)
    data = reader.read_numpy()
    im = Image.fromarray(data[0])
    im.save('../data/output/city_patches/png/washington/'+str(i)+'.png')




