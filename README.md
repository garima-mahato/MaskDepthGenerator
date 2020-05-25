# MaskDepthGenerator

**MaskDepthGenerator** takes an image and its background as input to generate depth and mask for the image. 

## Dataset

[Custom Dataset](https://drive.google.com/open?id=1zQTsYCo7_p-4u_3pgNCjd7dpGFRUZIVJ) was made and used for training the network. Details about it can be found [here](https://github.com/genigarus/DepthMaskDataset).


## Model Architecture

![Model Architecture](https://raw.githubusercontent.com/genigarus/MaskDepthGenerator/master/Assets/torchviz-model.png)

[**View full image of model here**](https://github.com/genigarus/MaskDepthGenerator/blob/master/Assets/torchviz-model.png)

This model is based on Encoder-Decoder architectural style. Pre-trained ResNet18 with first and last layer removed is taken as the backbone for the encoder. The Decoder consist of two branches - one for generating the mask, and the other for generating the depth. Each branch of the Decoder consist of [2, 2, 2, 2] block similar to ResNet18 backbone and each block in the decoder has a skip connection to the corresponding block in the encoder.

![](https://raw.githubusercontent.com/genigarus/MaskDepthGenerator/master/Assets/net_params.PNG)

**Total number of parameters: 16 million**

## Code Structure

[Link to API used for **MaskDepthGenerator**](https://github.com/genigarus/API)


## Code Profiling

Below is the GPU and memory usage of training code.

![](https://raw.githubusercontent.com/genigarus/MaskDepthGenerator/master/Assets/training_profiling1.PNG)

This is the link to [training code profiling](https://github.com/genigarus/MaskDepthGenerator/blob/master/train_code_lines_profiling.txt)

From this, we can see that 78% of the time is spent on loss calculation and batch sampling.


## Training

[Link to model training file](https://github.com/genigarus/MaskDepthGenerator/blob/master/ResMaskDepthGenerator.ipynb)

For training, I used SGD optimizer with 0.01 learning rate.

Initially, I trained the network with image size of 80x80. During this, I used L1 loss for depth prediction and BCELossWithLogits for mask prediction.


## Prediction


## Accuracy Metrics


