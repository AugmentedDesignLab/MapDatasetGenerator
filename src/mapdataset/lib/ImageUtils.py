import numpy as np
from PIL import Image

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