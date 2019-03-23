## U-net: Convolutional Networks for Biomedical Image Segmentation

[link to the authors' webpage](http://lmb.informatik.uni-freiburg.de/)

[link to the paper](https://arxiv.org/pdf/1505.04597.pdf)

The objective of the Unet architecture (May 2015) was good performance in biomedical applications. It often happens in that field that labeled datasets of thousands images (i.e. sufficient for training deep models) is simply unobtainable. Therefore, the paper's authors made a point of creating a model trainable on much fewer samples, on which they ran data augmentation algorithms. 

The architecture that they proposed is a fully-convolutional network. It does not have fully connected layers. It obtains good localization of pixel labels by applying particular multi-channel upsampling layers, and a U-shaped architecture (the expansive part - upsampling convolution - is more or less symmetric to the contracting - pooling - path).

### Data augmentation
Data augmentation was necessary for the model to obtain robustness (i.e. generalization power) with very few training samples available. The authors created new images with rotation and shift changes, as well as with deformations and gray scale variations. They also used so-called random-elastic deformations, which they claim to be the key to obtaining robustness. They also applied per-pixel displacements and, as implicit data augmentation - dropout layers at the end ot the contracting part.

### Training
To make the training time maximally efficient, the authors decided to reduce the batch size to one image. They decided for that step, because reducing the input image size would jeopardise accuracy. They also used very high momentum (0.99), and non-padded convolutional layers.
