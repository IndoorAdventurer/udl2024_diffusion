from torch.utils.data import Dataset
from torchvision.transforms import transforms


class NoLabelsWrapper(Dataset):
    """Constructs a dataset without labels from a dataset with labels."""

    def __init__(self, dataset: Dataset):
        super().__init__()
        self.ds = dataset
    
    def __getitem__(self, index):
        return self.ds.__getitem__(index)[0]
    
    def __len__(self):
        return self.ds.__len__()

def remove_dataset_labels(dataset: Dataset):
    """
    Args:
        dataset: a dataset that returns pairs (x, y).
    
    Returns:
        a dataset that only returns x, instead of (x, y).
    """
    return NoLabelsWrapper(dataset)

def default_img_transforms(num_channels: int = 3):
    """First calls ToTensor (should scale to range from 0 to 1), then re-scales
    to range between -1 and 1.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5] * num_channels, std=[.5] * num_channels),
    ])