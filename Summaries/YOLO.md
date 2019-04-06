## YOLO 

[link to the paper](https://arxiv.org/pdf/1506.02640.pdf)

YOLO (You Only Look Once) is an object detection system based on a single convolutial network that simultaneusly predicts multiple bounding boxes and class probabilities for those boxes. The aim of YOLO is to create a structure capable to be faster than Faster R-CNN to achieve the real-time object detection.

Yolo has reframed object detection as a single regression problem, straight from image pixels to bounding box coordinates and class probabilities. This unified model improves traditional methods in terms of speed due to its simple pipeline.
Instead of sliding windows and region proposal-based techniques, YOLO trains and test on full images so it implicitly encodes contextual information about classes as well as their appearance so it directly optimizes detection performance.

YOLO system divides the input image into an SxS grid. The grid cell responsible for detecting an object is the one in which the center of the object falls into. Each grid cell predicts 2 bounding boxes and confidence scores for those boxes and class probabilities for those bounding boxes. The system only predicts one set of class probabilities per grid cell.

It uses 24 convolutional layers that are simply based on 1x1 reduction layers followed by 3x3 convolutional layers and finally 2 fully connected layers (Fast YOLO uses 9 convolutional layers instead of 24). The system optimize for sum-squared error in the output of the model because it is easy to optimize.

The output of YOLO is an SxSx(Bx5+C) where SxS is the number of grid cells, B the number of bounding boxes (2), 5 is related to the x, y, width, height and confidence (center) parameters of the bounding box and C is the number of classes.

Limitations 
- Strong spatial constraints on bounding box predictions because each grid cell only predicts two boxes and can only have one class.
- Less accuracy than state of the art, specially on small objects.

Advantages
- Speed (real time video processing). 45 frames/second .
- Single regression problem 
- Is highly generalizable. Trains on full images.
- Simple pipline. Single regression problem.
