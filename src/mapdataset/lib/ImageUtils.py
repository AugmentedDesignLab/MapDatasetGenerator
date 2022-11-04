import numpy as np
from PIL import Image
import torch

class ImageUtils:

    @staticmethod
    def PILPatchToPILImg(patch):
        data = np.clip((patch + 1) / 2, 0, 1) * 255
        data = data.astype(np.uint8)
        return Image.fromarray(data)

    @staticmethod
    def TorchPatchToPILImg(patch):
        # TODO fix dimensions
        data = np.clip((patch + 1) / 2, 0, 1) * 255
        data = data.astype(np.uint8)
        return Image.fromarray(data)

    @staticmethod
    def TorchPatchToPILImg(t):
        return Image.fromarray(np.array(((t.squeeze(0).permute(1, 2, 0)+1)/2).clip(0, 1)*255).astype(np.uint8))

            
    @staticmethod
    def TorchPatchToPILImgGray(t):
        return Image.fromarray(np.array(((t.squeeze()+1)/2).clip(0, 1)*255).astype(np.uint8), "L")

    @staticmethod
    def TorchNpPatchToPILImgGray(t):
        # both tensor and numpy arrays have squeeze method
        return ImageUtils.TorchPatchToPILImgGray(t)

    @staticmethod
    def PILImgToTorch(im): # batch with 1. Standardized into (-1, 1). (h, w, c) - > (c, h, w)
        return torch.tensor(np.array(im.convert('RGB'))/255).permute(2, 0, 1).unsqueeze(0) * 2 - 1

    @staticmethod
    def PILGrayToTorch(im): # batch with 1. Standardized into (-1, 1). (h, w, c) - > (c, h, w) gray has no extra color channel
        return torch.tensor(np.array(im.convert('L'))/255).unsqueeze(0) * 2 - 1