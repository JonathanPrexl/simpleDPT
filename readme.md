# Simplified Implementation of DPT (Dense Prediction Transformer)

This repo contains a simplified implementation of the [original source code](https://github.com/isl-org/DPT) for the manuscript [Dense Prediction Transformers (DPT) - Ranftl et al., 2021](https://arxiv.org/abs/2103.13413).
The main difference is that this implementation takes away some flexibility for the sake of simplicity (fewer model sizes, etc.) and most importantly works without forward hooks.

## Repo Content

The file `vit_dpt.py` contains a ViT backbone implementation which outputs the intermediate features at four different layers (adjustable via the `num_layers_list` variable).
Those features get processed according to the DPT approach in three stages:

* Token extraction (`ViTTokenToSpatialFeature` class)
* Feature harmonization (`FeatureHarmonizer` class)
* Feature fusion (`FeatureMerging` class)

before a task-dependent head can be deployed.
All steps are wrapped in the forward pass of the `ViTDPT` module which can be found in `vit_dpt.py`.

Example configurations for a 16x16 and 8x8 patch size can be found under `configs/config_16x16.yaml` and `configs/config_8x8.yaml`.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
