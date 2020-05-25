# MaskDepthGenerator

**MaskDepthGenerator** takes an image and its background as input to generate depth and mask for the image. 

## Dataset

[Custom Dataset](https://drive.google.com/open?id=1zQTsYCo7_p-4u_3pgNCjd7dpGFRUZIVJ) was made and used for training the network. Details about it can be found [here](https://github.com/genigarus/DepthMaskDataset).


## Model Architecture

![Model Architecture](https://raw.githubusercontent.com/genigarus/MaskDepthGenerator/master/Assets/torchviz-model.png)

[**View full image of model here**](https://github.com/genigarus/MaskDepthGenerator/blob/master/Assets/torchviz-model.png)

This model is based on Encoder-Decoder architectural style. Pre-trained ResNet18 with first and last layer removed is taken as the backbone for the encoder. The Decoder consist of two branches - one for generating the mask, and the other for generating the depth. Each branch of the Decoder consist of [2, 2, 2, 2] block similar to ResNet18 backbone and each block in the decoder has a skip connection to the corresponding block in the encoder.

**Total number of parameters: 16 million**

## Code Structure

[Link to API used for **MaskDepthGenerator**](https://github.com/genigarus/API)


## Code Profiling


## Training


## Prediction


## Accuracy Metrics


