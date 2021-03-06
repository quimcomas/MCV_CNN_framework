## Faster R-CNN 

[link to the paper](https://arxiv.org/pdf/1506.01497.pdf)

The proposed network on this paper consist on a faster version of the R-CNN which reduces the computational time of object detection networks, called Faster R-CNN. 

The Faster R-CNN consist on two modules or networks. Basically, it is composed by a region proposal network (RPN) for generating region proposals and another network, Fast R-CNN, which use these region proposals to detect objects.

The key idea of RPN is rank region boxes, called anchors, and proposes the ones most likely containing objects.

In more detail, the FRCNN takes the feature maps from the CNN and passes them on the RPN where it uses a sliding window over these feature maps, and at each window, it generates k anchors boxes of different shapes and sizes. The RPN is formed by a classifier and a regressor. For each anchor, RPN predicts the probability that an anchor is an object (classifier) and position of the anchors to better fits the object (regressor).

Therefore, the main steps of the Faster R-CNN are:
-	Take an input image and pass it to the CNN network which returns feature maps output of the image. 
-	Apply RPN on these feature maps to obtain the object proposals.
-	Apply ROI polling layer to reduce all the proposals to the same size.
-	Finally, pass these proposals to a FC layer to classify and output the bounding boxes for objects.

To sum up, the authors of the paper present a RPNs for efficient and accurate region proposal generation. The advantage using RPN is that it shares full-image convolutional features with the detection network which produces nearly cost-free region proposals. The method enables a unified, deep-learning based object detection system to run at near real-time frame rates. The learned RPN also improves region proposal quality and thus the overall object detection accuracy.
