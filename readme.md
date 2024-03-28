# Papers
My implementations of the seminal papers of machine learning.

Inspired (to start) by https://github.com/hitorilabs/papers.

Will work through this while going through PRML.

This repo also serves as a devlog and notes.

**Table of Contents:**
1. [Initialization](#initialization) 
2. [ResNet](#resnet)
3. [Random Forest](#random-forest)
4. [Perceptron](#perceptron)

## Initialization
> pip install numpy mlx torch pandas matplotlib-pyplot 


## ResNet
* Residual network
* Very vertical

![ResBlock](resblock.png)

devlog:
* the shape after downsampling doesn't match the shape after the 2 conv layers (3x3 kernel, stride=2)
* the shapes are (100, 14, 14, 128) for the convs (56 divided by 2 twice) and the sequential returns (100, 7, 7, 128)

bugs:
* the input channel issue: an extra reshape inside the ``__call__`` messed the reshaping up

