import torch
from torch.utils.data import IterableDataset

from torchvision import transforms
from PIL import Image


class SingleImgDataset(IterableDataset):
    """
    Creates a dataset that returns a single image. These images:
    - are resized and cropped to shape [3 x img_size x img_size]
    - re-scaled to a range between -1 and 1.
    """

    def __init__(self, img_path: str, img_size: int):
        """
        Args:
            img_path: path to the image to select.
            img_size: will use this to create a square center crop (e.g. 64x64)
        """
        super().__init__()

        # Loading image and transforming:
        # - resize to [3 x img_size x img_size]
        # - scale pixels to range [-1, 1]
        img = Image.open(img_path)
        img: torch.Tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
        ])(img)

        self.img = img
    
    def __iter__(self):
        while True:
            yield self.img