from torchvision.models import resnet18, resnet34
from torch import nn

def discriminator_factory(in_channels: int, model_cls = resnet34):
    """Returns a resnet-based discriminator model to train the GAN.
    
    Args:
        in_channels: channels of input images.
        model_cls: the backbone model.
    """
    model = model_cls(weights=None)

    # Replace first conv:
    old_conv = model.conv1
    model.conv1 = nn.Conv2d(
        in_channels=in_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias,
    )

    # Replace final classification layer:
    old_fc = model.fc
    model.fc = nn.Linear(old_fc.in_features, 1, bias=(old_fc.bias != None))
                         
    return model