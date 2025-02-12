import torch
from torch.utils.data import IterableDataset


class PureNoiseDataset(IterableDataset):
    """Dataset made up out of random gaussian noise in the desired shape.
    """

    def __init__(self, img_shape: tuple[int]):
        """
        Args:
            img_shape: shape to use for generating random noise (e.g. CxHxW)
        """
        super().__init__()
        self.shape = img_shape
    
    def __iter__(self):
        while True:
            yield torch.randn(self.shape)