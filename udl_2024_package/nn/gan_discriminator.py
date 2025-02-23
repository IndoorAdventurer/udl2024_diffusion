from torch import nn


class CustomResNetBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, downsample: bool, kernel_size: int = 3):
        super(CustomResNetBlock, self).__init__()
        self.downsample = downsample
        
        self.skip_conv = None
        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, 1)
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding='same')
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding='same')
    
    def forward(self, x):
        skip = nn.functional.avg_pool2d(x, 2) if self.downsample else x
        if self.skip_conv is not None:
            skip = self.skip_conv(skip)
        
        x = self.conv1(x)
        # TODO: maybe batchnorm
        x = nn.functional.relu(x)

        x = self.conv2(x)
        # TODO: maybe batchnorm
        x = nn.functional.relu(x)

        if self.downsample:
            x = nn.functional.avg_pool2d(x, 2)

        return x + skip

class SimpleDiscriminator(nn.Module):
    """Discriminator model for GAN-based learning."""

    def __init__(self,
                 in_channels: int = 3,
                 channel_list: list[int] = [128] * 4,
                 kernel_list: list[int] = [3] * 4,
                 downsample_list: list[bool] = [True] * 2 + [False] * 2
        ):
        """
        Args:
            in_channels: number of channels of input image
            channel_list: for each resnet block, the number of channels
            kernel_list: for each resnet block, the conv2d kernel size
            downsample_list: for each resnet block, if it downsamples or not
        """
        super(SimpleDiscriminator, self).__init__()

        resnet_layers = []
        for ch, k, ds in zip(channel_list, kernel_list, downsample_list):
            resnet_layers.append(CustomResNetBlock(in_channels, ch, ds, k))
            in_channels = ch
        self.resnet_layers = nn.Sequential(*resnet_layers)

        self.out = nn.Linear(in_channels, 1)
    
    def forward(self, x):
        x = self.resnet_layers(x)
        x = x.view(*x.shape[:2], -1).mean(-1, False)
        x = self.out(x)
        return x