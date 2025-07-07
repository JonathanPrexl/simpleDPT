import torch
import torch.nn as nn
from torch.nn import functional as F

class ResidualConvUnit(nn.Module):
    """Residual convolution module."""

    def __init__(self, num_cnn_features, activation, bn):

        super().__init__()

        self.bn = bn

        self.groups = 1

        self.conv1 = nn.Conv2d(
            num_cnn_features,
            num_cnn_features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.bn,
            groups=self.groups,
        )

        self.conv2 = nn.Conv2d(
            num_cnn_features,
            num_cnn_features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.bn,
            groups=self.groups,
        )

        if self.bn == True:
            self.bn1 = nn.BatchNorm2d(num_cnn_features)
            self.bn2 = nn.BatchNorm2d(num_cnn_features)


        if activation == "relu":
            self.activation = nn.ReLU(inplace=False)
        else:
            raise NotImplementedError(
                f"Activation function {activation} is not implemented."
            )

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):

        out = self.activation(x)
        out = self.conv1(out)
        if self.bn == True:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn == True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)

class FeatureMerging(nn.Module):

    def __init__(
        self,
        num_cnn_features,
        activation,
        deconv=False,
        bn=False,
        align_corners=True,
    ):

        super(FeatureMerging, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups = 1

        self.out_conv = nn.Conv2d(
            num_cnn_features,
            num_cnn_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            groups=1,
        )

        self.resConfUnit1 = ResidualConvUnit(num_cnn_features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(num_cnn_features, activation, bn)

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs):
        
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=self.align_corners
        )

        output = self.out_conv(output)

        return output
    

if __name__ == "__main__":
    
    from torchinfo import summary
    from vit_tokens_to_spatial_features import ViTTokenToSpatialFeature
    from modules.feature_harmonizer import ChannelHarmonizer

    vit_features=768
    num_cnn_features_after_extraction=[96, 192, 384, 768]
    image_size = 384
    patch_size = 16
    num_tokens = (image_size//patch_size)**2

    tokens = torch.randn(1, num_tokens, vit_features)  # Example input tensor

    # ==================================================
    # get extracted spatially aligned featuresmaps from the ViT tokens 
    module = ViTTokenToSpatialFeature(
        image_size=image_size,
        patch_size=patch_size,
        vit_embedding_dim=vit_features,
        num_cnn_features_after_extraction=num_cnn_features_after_extraction
    )

    feature_1, feature_2, feature_3, feature_4 = module(*[tokens for i in range(4)])

    # ==================================================
    # now harmonize the number of channels for each layer

    num_cnn_features = 256  # Define the number of CNN features you want to harmonize to

    module = ChannelHarmonizer(
        num_cnn_features_after_extraction=num_cnn_features_after_extraction,
        num_cnn_features=num_cnn_features,
        groups=1)

    feature_1, feature_2, feature_3, feature_4 = module(
        feature_1, feature_2, feature_3, feature_4
    )

    print(feature_1.shape, feature_2.shape, feature_3.shape, feature_4.shape)

    # merge the features from differnt layers
    module = FeatureMerging(
        num_cnn_features=num_cnn_features,
        activation="relu",
        deconv=False,
        bn=False,
        align_corners=True)
    
    feature_4_upsampled = module(feature_4)
    feature_3_upsampled = module(feature_4_upsampled, feature_3)
    feature_2_upsampled = module(feature_3_upsampled, feature_2)
    feature_1_upsampled = module(feature_2_upsampled, feature_1)

    print(feature_1_upsampled.shape)

    pass 