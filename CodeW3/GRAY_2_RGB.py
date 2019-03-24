import cv2
import os.path
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

path_predictions = "/home/grupo03/M5/Code/test_fastnet/testFastNetMaxZUZA2/predictions"
output_path = "/home/grupo03/M5/Code/test_fastnet/testFastNetMaxZUZA2/predictions_RGB"

for image in os.listdir(path_predictions):
    print(image)
    image_RGB = cv2.imread(os.path.join(path_predictions, image), cv2.COLOR_GRAY2RGB)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    plt.imshow(image_RGB)

    plt.savefig(output_path+image)


