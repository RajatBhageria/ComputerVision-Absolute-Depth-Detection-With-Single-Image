import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import cv2
from BoundingBoxLabeling import BoundingBoxLabeling
from linreg_closedform import LinearRegressionClosedForm as LinearRegression
from PIL import Image

# Load Image and Depth Data
depths = np.load('../data/nyu_dataset_depths.npy')
images = np.load('../data/nyu_dataset_images.npy')

# Create bounding boxes to be labeled
## Starting point
i = int(raw_input("Enter img#: "))

while(1):
    # get get the rbg image
    imgi = images[:,:,:,i]
    while (1) :
        # bbox size [k,5] where n is image number, k is num of objects in each image
        # last dimension has x, y, height, width, depth of each bbox in image i
        bbox = BoundingBoxLabeling(imgi, depths[:,:,i], True, imageNum = i)
        entered = raw_input("See this image again (n/y): ")
        if entered != 'y':
            break
        pilimg = Image.fromarray(imgi, 'RGB')
        pilimg.show()

    i = i+1;
