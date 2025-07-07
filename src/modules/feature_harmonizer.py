import torch
import torch.nn as nn

class FeatureHarmonizer(nn.Module):

    """
    Takes the extracted features that have been processed by the
    ViTTokenToSpatialFeature module and harmonizes the number of channels 
    for each layer.

    """

    def __init__(self, 
                 num_cnn_features_after_extraction: list,
                 num_cnn_features: int,
                 groups: int,
                 ):
        super().__init__()

        self.process_layer_1 = nn.Conv2d(
            num_cnn_features_after_extraction[0],
            num_cnn_features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            groups=groups,
        )

        self.process_layer_2 = nn.Conv2d(
            num_cnn_features_after_extraction[1],
            num_cnn_features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            groups=groups,
        )

        self.process_layer_3 = nn.Conv2d(
            num_cnn_features_after_extraction[2],
            num_cnn_features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            groups=groups,
        )

        self.process_layer_4 = nn.Conv2d(
            num_cnn_features_after_extraction[3],
            num_cnn_features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            groups=groups,
        )

    def forward(self, layer_1, layer_2, layer_3, layer_4):

        layer_1_rn = self.process_layer_1(layer_1)
        layer_2_rn = self.process_layer_2(layer_2)
        layer_3_rn = self.process_layer_3(layer_3)
        layer_4_rn = self.process_layer_4(layer_4)

        return layer_1_rn, layer_2_rn, layer_3_rn, layer_4_rn
    


if __name__ == "__main__":
    
    from torchinfo import summary
    from vit_tokens_to_spatial_features import ViTTokenToSpatialFeature

    vit_features=768
    num_cnn_features_after_extraction=[96, 192, 384, 768]
    imagesize = 128
    patchsize = 8
    num_tokens = (imagesize//patchsize)**2

    tokens = torch.randn(1, num_tokens, vit_features)  # Example input tensor

    # ==================================================
    # get extracted spatially aligned featuresmaps from the ViT tokens 

    module = ViTTokenToSpatialFeature(
        imagesize=imagesize,
        patchsize=patchsize,
        vit_embedding_dim=vit_features,
        num_cnn_features_after_extraction=num_cnn_features_after_extraction
    )

    feature_1, feature_2, feature_3, feature_4 = module(tokens)

    print(feature_1.shape, feature_2.shape, feature_3.shape, feature_4.shape)


    # ==================================================
    # now harmonize the number of channels for each layer

    num_cnn_features = 256  # Define the number of CNN features you want to harmonize to

    module = ChannelHarmonizer(
        num_cnn_features_after_extraction=num_cnn_features_after_extraction,
        num_cnn_features=num_cnn_features,
        groups=1,
        expand_feature_number=False
    )

    feature_1, feature_2, feature_3, feature_4 = module(
        feature_1, feature_2, feature_3, feature_4
    )

    print(feature_1.shape, feature_2.shape, feature_3.shape, feature_4.shape)


    pass