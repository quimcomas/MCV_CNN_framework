## FCN: Fully Convolutional Networks for Semantic Segmentation

[link to the paper](https://arxiv.org/pdf/1411.4038.pdf)

To divide the images in visually semantically meaningful patches is very important for a lot of computer vision processes, for example, for autonomous vehicles scene understanding. This is called Semantic Segmentation and it was traditionally applied without using Neural Networks., dividing the images in visually similar patches.
In this paper, J. Long, E. Shelhamer and T. Darrell present the Fully Convolutional Networks for Semantic Segmentation.

This method take images of arbitrary size and produce an image of the same output size. It divides the image in semantically meaningful patches, obtaining an array of classes for each pixel, in wich the higher probability determines the class of each pixel. This method reinterpretes classification networks in order to use it for Semantic Segmentation, for example, VGG, AlexNet or GoogleNet can be used as convolution and trained on ImageNet.

The FCN takes the three-dimensional array of size (h x w x d), where h and w are spatial dimensions and d the feature dimension. While a general deep net computes a general nonlinear function, this type of nets computes a nonlinear filter, wich they call deep filter or fully convolutional network.
So, they transform the fully connected layers into a convolutional layers in order to output a heatmap (pixelwise prediction) of the image to segment it in meaningful patches.

In order to obtain this pixelwise prediction, this last convolutional layer has to be upsampled, using backwards convolution (sometimes called deconvolution). For upsampling, it is necessary to play with the semantic precision and spatial precision trade-off.
It is necessary to combine coarse, high layer information with fine, low layer information to obtain a finer details and precision with good semantic information. They do that combining early pooling layers to the prediction. For example, using FCN-8s means that it is combined the pool4 layer at stride 16 and also the pool3 at stride 8, and that provides very good precision.

Extending these clasification networks to segmentation, and improving the architecture with multi-resulution layer combinations dramatically improves the state-of-the-art while simultaneausly simplifying and speeding up learning and inference.
