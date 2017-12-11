import numpy as np
import cv2
from find_BB_and_depthTest import find_BB_and_depth as find_BB_and_depth
# import load_mat_to_python
from linreg_closedform import LinearRegressionClosedForm as LinearRegression
from PIL import Image
import sys

# Part 0: Loading the data with depth from matlab to python

# convert the matlab file to python
# load_mat_to_python()

# DONE

# Part 1: Loading image and associated depth data into python

# load all the data
depths = np.load('data/nyu_dataset_depths.npy')
images = np.load('data/nyu_dataset_images.npy')
# labels = np.load('nyu_dataset_labels.npy')
# names = np.load('nyu_dataset_names.npy')
# scenes = np.load('nyu_dataset_scenes.npy')

# Part 2: Import labels n by 4 (img #, bb#, lab_h, lab_w)

labels = np.loadtxt('data/ImageLabels.dat', delimiter=',')
n, d = labels.shape

# array to hold (img#, bb#, lab_h, lab_w, x, y, h, w, d, img_h, img_w)
imageLabels = np.zeros((n,11))
imageLabels[:,0:4] = labels

# Part 3: Create bounding boxes for our training images


i = raw_input("Enter img#: ")
i = int(i)


print("Image entered: " + str(i))

while(1):

    print("Current Image: " + str(i))
    imgi = images[:,:,:,i]
    h,w,c = imgi.shape

    #show the image


    # bbox size [k,5] where n is image number, k is num of objects in each image
    # last dimension has x, y, height, width, depth of each bbox in image i

    while (1) :
        # bbox size [k,5] where n is image number, k is num of objects in each image
        # last dimension has x, y, height, width, depth of each bbox in image i
        bbox = find_BB_and_depth(imgi, depths[:,:,i], True)
        entered = raw_input("See this image again (n/y): ")
        if entered != 'y':
            break
        pilimg = Image.fromarray(imgi, 'RGB')
        pilimg.show()

    i = i+1;

    # # add to the allBBoxes matrix
    # k = int(imageLabels[i,1])
    #
    # #add the bbox values to the imageLabel
    # imageLabels[i, 4:9] = bbox[k]
    #
    # #add the height width of the image to the imageLabels
    # imageLabels[i, 9:11] = (h, w)

    # add to the allBBoxes matrix

# Part 4: Aggregate training data

train_height = imageLabels[:, [6, 8, 9]]
train_width = imageLabels[:, [7, 8, 10]]

label_height = imageLabels[:, 2]
label_width = imageLabels[:, 3]

# Part 5: Fit a Linear Regression with training data

# do training on linear regression
linreg_x = LinearRegression(regLambda = 1E-8)
linreg_y = LinearRegression(regLambda = 1E-8)
linreg_x.fit(train_height, label_height)
linreg_y.fit(train_width, label_width)

# Part 6: Test with new data

unlabeled = np.loadtxt('data/ImageUnLabeled.dat', delimiter=',')
n, d = unlabeled.shape

# array to hold (img#, bb#, null, null, x, y, h, w, d, img_h, img_w)
imageUnLabeled = np.zero(n,11)
imageUnLabeled[:,0:2] = unlabeled

# Part 7: Create bounding boxes for our testing images

# for i in range(n):
while (1):
    i = raw_input("Input img#")
    imgNum = imageUnLabeled[int(i),0]
    imgi = images[:,:,:,imgNum]
    h,w,c = imgi.shape
    pilimg = Image.fromarray(imgi, 'RGB')
    pilimg.show()

    # bbox size [k,5] where n is image number, k is num of objects in each image
    # last dimension has x, y, height, width, depth of each bbox in image i
    bbox = find_BB_and_depth(imgi, depths[i])

    # add to the allBBoxes matrix
    k = imageUnLabeled[i,1]
    imageUnLabeled[i, 4:9] = bbox[k]
    imageUnLabeled[i, 9:11] = (h, w)

test_height = imageUnLabeled[:, [6, 8, 9]]
test_width = imageUnLabeled[:, [7, 8, 10]]

# predict sizes for k objects in n images using linreg
result_height = linreg_x.predict(test_height)
result_width = linreg_y.predict(test_width)

# do training on neural nets

# TODO: Neural Network

# predict sizes for k objects in n images using neuralnets
