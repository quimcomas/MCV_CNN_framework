## VGG

#### Very Deep Convolutional Networks for Large-scale Image Recognition

The Convolution Neural Network proposed by the Visual Geometry Group (VGG) of the University of Oxford won the first place localisation and the second place for image recognition in the ImageNet Challenge 2014 (ILSVRC 2014).  Respect to the image recognition task the VGG obtained a 7.3% in top-5 error (after the publication they obtained a 6.8 % in top-5 error), quite close to GoogLenet which obtained a 6.7% in top-5 error.

The main idea of VGG is that fixing some parameters and increasing the depth of the network by adding more convolutional layers can be beneficial for the classification accuracy.

Respect to the VGG architecture, which is similar to the Alexnet, introduced a small filter sizes (3x3), using 1 pixel for stride and padding. Also it used 5 non-overlapping max-pooling of 2x2 pixel window. The network end with 3 fully connected layers, two layers with 4096 channels and the last layer of 1000 channels for classify the different classes using a soft-max layer. As in Alexnet network the authors used the Relu activation in all hidden layers and they added dropout and data augmentation.

In the training process they used a mini-batch gradient descent with a batch size of 256, momentum 0.9 and the learning rate was set to 10^-2. Also the authors proposed two approaches for the weights initialisation. On one hand, the weights were initialized by training a shallower network and transferring the weights to the larger network. On the other hand, the used the Glorot random initialisation.

About the classification experiments they used: 

-	Single scale evaluation where the test images were evaluated at a fixed size. The best results were 25.5 % in top-1 error and 8 % in top-5 error.
-	Multi-scale evaluation where the model was run over several rescaled versions of each test image. The best results were 24.8 % in top-1 error and 7.5 % in top-5 error.    
-	Multi-crop evaluation where the model was run over multiple crops of each test image. The best results were 24.4 % in top-1 error and 7.1 % in top-5 error.    

Finally, they combined the output of several models by averaging their soft-max class posteriors. In the case of the D and E networks, they achieved a 23.7 in top-1 error and 6.8% in top-5 error.   

