import torch
import torch.nn as nn
from torch.nn import functional as F

class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        x = x.transpose(self.dim0, self.dim1)
        return x
    
class ViTTokenToSpatialFeature(nn.Module):

    """
    Adapted from https://github.com/isl-org/DPT
    Takes the imagetokens and re-aligns them to match the spatial layout
    of the image data.

    Args:
    image_size (int): Size of the input image (e.g., 384 for 384x384 images).
    patch_size (int): Size of the patches (e.g., 16 for 16x16 patches).
    vit_embedding_dim (int): Dimension of the ViT embeddings (e.g., 768).
    num_cnn_features (list): List of CNN feature dimensions for each layer
                            (e.g., [96, 192, 384, 768]).
   
    Input image with size 384 and 16 patch size (24*24 patches) will lead to:
    num_cnn_features[0] x 96 x 96 (4x upsampled)
    num_cnn_features[1] x 48 x 48 (2x upsampled)
    num_cnn_features[2] x 24 x 24 (no upsampled)
    num_cnn_features[3] x 12 x 12 (2x downsampled)

    Input image with size 128 and patch size 8 (16*16 patches) will lead to:
    num_cnn_features[0] x 64 x 64 (4x upsampled)
    num_cnn_features[1] x 32 x 32 (2x upsampled)
    num_cnn_features[2] x 16 x 16 (no upsampled)
    num_cnn_features[3] x 8 x 8 (2x downsampled)


    The unflatten layers are used to reshape the input tokens into a 2D grid
    that matches the spatial layout of the image patches. As for the debugging case
    in 
    """

    def __init__(self,
                 image_size: int,
                 patch_size: int,
                 vit_embedding_dim: int,  
                 num_cnn_features_after_extraction: list,               
                 ):
        super().__init__()

        self.process_layer_1 = nn.Sequential(
            Transpose(1, 2),
            nn.Unflatten(2, torch.Size([image_size // patch_size, image_size // patch_size])),
            nn.Conv2d(
                in_channels=vit_embedding_dim,
                out_channels=num_cnn_features_after_extraction[0],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.ConvTranspose2d(
                in_channels=num_cnn_features_after_extraction[0],
                out_channels=num_cnn_features_after_extraction[0],
                kernel_size=4,
                stride=4,
                padding=0,
                bias=True,
                dilation=1,
                groups=1,
            ),
        )

        self.process_layer_2 = nn.Sequential(
            Transpose(1, 2),
            nn.Unflatten(2, torch.Size([image_size // patch_size, image_size // patch_size])),
            nn.Conv2d(
                in_channels=vit_embedding_dim,
                out_channels=num_cnn_features_after_extraction[1],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.ConvTranspose2d(
                in_channels=num_cnn_features_after_extraction[1],
                out_channels=num_cnn_features_after_extraction[1],
                kernel_size=2,
                stride=2,
                padding=0,
                bias=True,
                dilation=1,
                groups=1,
            ),
        )

        self.process_layer_3 = nn.Sequential(
            Transpose(1, 2),
            nn.Unflatten(2, torch.Size([image_size // patch_size, image_size // patch_size])),
            nn.Conv2d(
                in_channels=vit_embedding_dim,
                out_channels=num_cnn_features_after_extraction[2],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

        self.process_layer_4 = nn.Sequential(
            Transpose(1, 2),
            nn.Unflatten(2, torch.Size([image_size // patch_size, image_size // patch_size])),
            nn.Conv2d(
                in_channels=vit_embedding_dim,
                out_channels=num_cnn_features_after_extraction[3],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.Conv2d(
                in_channels=num_cnn_features_after_extraction[3],
                out_channels=num_cnn_features_after_extraction[3],
                kernel_size=3,
                stride=2,
                padding=1,
            ),
        )

    def forward(self, layer_1, layer_2, layer_3, layer_4):

        layer_1 = self.process_layer_1(layer_1)
        layer_2 = self.process_layer_2(layer_2)
        layer_3 = self.process_layer_3(layer_3)
        layer_4 = self.process_layer_4(layer_4)

        return layer_1, layer_2, layer_3, layer_4




if __name__ == "__main__":
    
    from torchinfo import summary

    vit_features=768
    num_cnn_features_after_extraction=[96, 192, 384, 768]
    image_size = 128
    patch_size = 8
    num_tokens = (image_size//patch_size)**2

    tokens = torch.randn(1, num_tokens, vit_features)  # Example input tensor

    module = ViTTokenToSpatialFeature(
        image_size=image_size,
        patch_size=patch_size,
        vit_embedding_dim=vit_features,
        num_cnn_features_after_extraction=num_cnn_features_after_extraction
    )

    summary(module.process_layer_1, input_size=tokens.shape)
    summary(module.process_layer_2, input_size=tokens.shape)
    summary(module.process_layer_3, input_size=tokens.shape)
    summary(module.process_layer_4, input_size=tokens.shape)

    pass
