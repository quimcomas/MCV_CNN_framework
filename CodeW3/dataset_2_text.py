"""
Usage:
  main.py <dataset_path> <outputFile_path>
  main.py -h | --help
Options:
  <dataset_path>    Specify the full path to the dataset directory
  <outputFile_path> Directory to output the .txt files with all the filenames and gt data.
Example:
  python2 datasetToTxt.py /home/mcv/datasets/M5/classification/KITTI/ KITTI_txt

"""

# Imports
import os
import sys
import shutil
import glob
import numpy as np
from docopt import docopt


# Create a file with all the images as paths
def from_images_to_txt(directory, outDirectory):
    directory = os.path.join(directory)
    out_file = outDirectory + "_images.txt"
    out_file_label = outDirectory + "_labels.txt"

    folders = []

    for subdir in sorted(os.listdir(directory)):
        if os.path.isdir(os.path.join(directory, subdir)):
            folders.append(subdir)
    print(folders)

    filenames = []
    y = []

    # Get filenames

    for subdir in folders:

        if subdir == 'images':
            subpath = os.path.join(directory, subdir)
            for fname in os.listdir(subpath):
                filenames.append(os.path.join(subpath, fname))

        if subdir == 'masks':
            subpath1 = os.path.join(directory, subdir)
            for fname1 in os.listdir(subpath1):
                y.append(os.path.join(subpath1, fname1))

    print("   Saving file: " + out_file)
    outfile = open(out_file, "w")
    out_file_label = open(out_file_label, "w")

    for item in filenames:
        outfile.write("%s\n" % item)

    for label in y:
        out_file_label.write("%s\n" % label)


def main():
    if len(sys.argv) > 1:
        args = docopt(__doc__)

        directory = args['<dataset_path>']
        outDirectory = args['<outputFile_path>']

        from_images_to_txt(directory+'/train', outDirectory+'/train')
        from_images_to_txt(directory+'/valid', outDirectory+'/valid')
        from_images_to_txt(directory+'/test', outDirectory+'/test')

    else:
        print ('Not enough arguments')
        exit()

# Entry point of the script
if __name__ == "__main__":
    main()
