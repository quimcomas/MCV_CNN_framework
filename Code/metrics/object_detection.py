import os
import numpy as np

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def Compute_kitti_AP(scores_file):
    filename = os.path.join(scores_file)
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            vLines = f.readlines()
            vLines = np.asarray([[float(number) for number in line.rstrip().split(' ')] for line in vLines])
        mean_scores = np.mean(vLines, axis=1)
        return mean_scores
    return np.zeros(3, dtype=np.float32)