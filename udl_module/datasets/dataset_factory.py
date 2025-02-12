from torchvision.datasets import VisionDataset, CIFAR10
from torchvision.transforms import transforms


def dataset_factory(dataset_cls: type[VisionDataset] = CIFAR10, path: str = "./datasets"):
    """Creates a dataset object with all the appropriate transforms for doing
    diffusion, given the class type of that dataset.

    Args:
        dataset_cls: class of the dataset to use, like CIFAR or MNIST.
        path: folder to download datasets into.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]),
    ])

    # Wrapper that discards the labels:
    class NoLabelsDS(dataset_cls):
        def __getitem__(self, index):
            img, _ = super().__getitem__(index)
            return img

    return (
        NoLabelsDS(root=path, transform=transform, download=True, train=True),
        NoLabelsDS(root=path, transform=transform, download=True, train=False)
    )