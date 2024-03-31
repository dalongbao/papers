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
* First paper fully implemented by me from reading the paper!

![ResBlock](/images/resblock.png)

Devlog/improvements:
* change line 273 to mx.array, changing the effects afterward

Bug log:
* the input channel issue: an extra reshape inside the ``__call__`` messed the reshaping up
* residual operation shape issue bug: the stride was wrong, downsampled a bit too much
* hard coded the batch\_size and matrix reshaping; changed it to the variable names and was fixed
