from diffusers import UNet2DModel
from torch import  nn

# Default architecture copied from:
#   https://huggingface.co/google/ddpm-cifar10-32/blob/main/config.json
def unet_factory(
    img_size: int = 32,
    img_channels: int = 3,
    block_out_channels: list[int] = [128, 256, 256, 256],
    layers_per_block: int = 2,
) -> nn.Module:
    """Returns a U-Net PyTorch Module from arguments.

    Args:
        img_size: width/height of images. Must be a multiple of 2.
        img_channels: number of channels per image. RGB is 3, but MNIST is 1.
        block_out_channels: output channels per block. Must be 4 I think.
        layers_per_block: number of layers per block.
    """
    class DiffusersUnet(UNet2DModel):
        def forward(self, *args, **kwargs):
            return super().forward(*args, **kwargs).sample

    return DiffusersUnet(
        sample_size = img_size,
        in_channels = img_channels,
        out_channels = img_channels,
        center_input_sample = False,
        time_embedding_type = 'positional',
        # time_embedding_dim: Optional[int] = None,
        freq_shift = 1,
        flip_sin_to_cos = False,
        down_block_types = ['DownBlock2D',
                            'AttnDownBlock2D',
                            'DownBlock2D',
                            'DownBlock2D'],
        up_block_types = ['UpBlock2D', 'UpBlock2D', 'AttnUpBlock2D', 'UpBlock2D'],
        block_out_channels = block_out_channels,
        layers_per_block = layers_per_block,
        mid_block_scale_factor = 1,
        downsample_padding = 0,
        # downsample_type: str = "conv",
        # upsample_type: str = "conv",
        # dropout: float = 0.0,
        act_fn = 'silu',
        attention_head_dim = None,
        norm_num_groups = 32,
        # attn_norm_num_groups: Optional[int] = None,
        norm_eps = 1e-06,
        # resnet_time_scale_shift: str = "default",
        # add_attention: bool = True,
        # class_embed_type: Optional[str] = None,
        # num_class_embeds: Optional[int] = None,
        # num_train_timesteps: Optional[int] = None,
    )