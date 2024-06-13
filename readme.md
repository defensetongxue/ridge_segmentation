# ROP Lesion Segmentation
[官中版](./说明.md)

This is the official repository for ROP lesion segmentation. Since the annotations are generated based on the [algorithm](./util/ridge_diffusion.py), this repository generates pixel-level segmentation labels, but we perform image-level testing when validating and testing.

`cleansing.py` will generate the corresponding labels based on point annotations and split the original images according to the settings in the config. By default, we use a patch size of 400 and a stride of 200 for splitting.

`train.py` and `train_vit.py` are used for training models with different networks. The latter is suitable for TransUNet, which requires specific input dimensions and utilizes global information, so the entire image is inputted. The former is for other CNNs such as HRNet and UNet.

`test.py` and `test_vit.py` are used for testing.
