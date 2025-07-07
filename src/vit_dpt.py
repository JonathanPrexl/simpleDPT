import torch
import torch.nn as nn
from torch.nn import functional as F

from modules.vit_tokens_to_spatial_features import ViTTokenToSpatialFeature
from modules.feature_harmonizer import FeatureHarmonizer


class TokenReshape(nn.Module):
    """Reshape ViT tokens to spatial features."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # flatten the spatial dimensions
        # order of the spatial information (lets asume 8x8 grid as in 128 image size
        # and 16 patch size) is here:
        # (x0 y0) (x1 y0) (x2 y0) ... (x7 y7)
        # this gets importent when transforming the tokens to spatial features
        # in the ViTTokenToSpatialFeature module
        x = x.flatten(2).transpose(1, 2)
        return x
    

class ViTBackbone(nn.Module):

    def __init__(self,
                 image_size: int,
                 number_of_channels: int,
                 patch_size: int,
                 num_layers_list: list,
                 vit_embedding_dim: int,
                 heads: int):
        
        super().__init__()

        assert len(num_layers_list) == 4, "num_layers_list must have exactly 4 elements since DPT implementation relies on it."
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size."
        assert vit_embedding_dim % heads == 0, "vit_embedding_dim must be divisible by heads."

        # patch embedding
        self.patch_embedding = nn.Conv2d(  
            in_channels=number_of_channels,
            out_channels=768,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # learnable positional encodings
        self.positional_embedding = nn.Parameter(
            torch.randn(1, (image_size // patch_size)**2, vit_embedding_dim)
        )

        self.reshape = TokenReshape()

        # four stages of transformer blocks
        # so no hook implementaion is needed
        transformer_layer = nn.TransformerEncoderLayer(d_model=vit_embedding_dim, nhead=heads, batch_first=True)
        self.transformer_block_1 = nn.TransformerEncoder(transformer_layer, num_layers=num_layers_list[0])
        self.transformer_block_2 = nn.TransformerEncoder(transformer_layer, num_layers=num_layers_list[1])
        self.transformer_block_3 = nn.TransformerEncoder(transformer_layer, num_layers=num_layers_list[2])
        self.transformer_block_4 = nn.TransformerEncoder(transformer_layer, num_layers=num_layers_list[3])

    def forward(self, x):

        # patch embedding
        x = self.patch_embedding(x)

        # reshape to tokens
        x = self.reshape(x)

        # add positional embedding
        x = x + self.positional_embedding

        # transformer blocks
        # no need to use hooks when defining seperate blocks
        features_1 = self.transformer_block_1(x)
        features_2 = self.transformer_block_2(features_1)
        features_3 = self.transformer_block_3(features_2)
        features_4 = self.transformer_block_4(features_3)

        return features_1, features_2, features_3, features_4

class Interpolate(nn.Module):
    """Interpolation module."""

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.

        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )

        return x
    
class DPTHead(nn.Module):

    def __init__(self, num_cnn_features, output_channels, upsample):
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv2d(num_cnn_features, num_cnn_features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True) if upsample else nn.ReLU(True),
            nn.Conv2d(num_cnn_features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, output_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.head(x)

class ViTDPT(nn.Module):

    def __init__(self,
                 vit_backbone,
                 token_extractor,
                 feature_harmonizer,
                 feature_merger,
                 prediction_head):
        super().__init__()

        self.vit_backbone = vit_backbone
        self.token_extractor = token_extractor
        self.feature_harmonizer = feature_harmonizer
        self.feature_merger = feature_merger
        self.prediction_head = prediction_head

    def forward(self, x):

        # Get ViT features
        features_1, features_2, features_3, features_4 = self.vit_backbone(x)

        # Extract spatially aligned feature maps from ViT tokens
        features_1, features_2, features_3, features_4 = self.token_extractor(
            features_1, features_2, features_3, features_4
        )

        # Harmonize thne number of channels for each layer
        features_1, features_2, features_3, features_4 = self.feature_harmonizer(
            features_1, features_2, features_3, features_4
        )

        # Merge the features from different layers
        features_4_upsampled = self.feature_merger(features_4)
        features_3_upsampled = self.feature_merger(features_4_upsampled, features_3)
        features_2_upsampled = self.feature_merger(features_3_upsampled, features_2)
        features_1_upsampled = self.feature_merger(features_2_upsampled, features_1)

        # apply prediction head
        output = self.prediction_head(features_1_upsampled)

        return output


if __name__ == "__main__":
    
    from torchinfo import summary
    import hydra
    from omegaconf import DictConfig


    @hydra.main(version_base=None, config_path="configs", config_name="config_8x8.yaml")
    def main(cfg: DictConfig):
        
        model = hydra.utils.instantiate(cfg.model)
        
        # Print model summary
        summary(model, input_size=(1, 3, 128, 128), col_names=["input_size", "output_size", "num_params", "trainable"], depth=2)
        
        # Test the model with a dummy input
        dummy_input = torch.randn(1, 3, 128, 128)
        output = model(dummy_input)
        print("Output shape:", output.shape)

    main()


