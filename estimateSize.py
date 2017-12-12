import numpy as np
import cv2
import csv
from find_BB_and_depth import find_BB_and_depth
# import load_mat_to_python
from linreg_closedform import LinearRegressionClosedForm as LinearRegression
from PIL import Image
from deepNeuralNet import runDNN
import sys


def estimateSize():

    # Part 0: Loading the data with depth from matlab to python

    # convert the matlab file to python
    # load_mat_to_python()

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
    imageLabels = np.zeros((n, 11))
    imageLabels[:, 0:4] = labels

    # Part 3: Create bounding boxes for our training images
    for i in range(n):
        # print("Testing on: " + str(imageLabels[i]))
        imgNum = int(imageLabels[i, 0])
        imgi = images[:, :, :, imgNum]
        h, w, c = imgi.shape

        # show the image
        # pilimg = Image.fromarray(imgi, 'RGB')
        # pilimg.show()

        # bbox size [k,5] where n is image number, k is num of objects in each image
        # last dimension has x, y, height, width, depth of each bbox in image i
        bbox = find_BB_and_depth(imgi, depths[:, :, imgNum], False)

        # add to the allBBoxes matrix
        k = int(imageLabels[i, 1])

        # add the bbox values to the imageLabel
        imageLabels[i, 4:9] = bbox[k]

        # add the height width of the image to the imageLabels
        imageLabels[i, 9:11] = (h, w)

    # Part 4: Aggregate training data

    # X train- training height and widths

    train_height = (imageLabels[:, 9] / 2 -
                    imageLabels[:, 6]) * imageLabels[:, 8]
    train_width = (imageLabels[:, 10] / 2 -
                   imageLabels[:, 7]) * imageLabels[:, 8]

                   # Old features
    # train_height = imageLabels[:, [6, 8, 9]]
    # train_width = imageLabels[:, [7, 8, 10]]

    # Y train- training heights and widths
    label_height = imageLabels[:, 2]
    label_width = imageLabels[:, 3]

    # Part 5: Generate the test data
    # image number, bouning box number
    unlabeledWithDescription = np.loadtxt(
        'data/ImageUnLabeled.dat', delimiter=',', usecols=(0, 1))
    n, c = unlabeledWithDescription.shape

    # array to hold (img#, bb#, null, null, x, y, h, w, d, img_h, img_w)
    imageUnLabeled = np.zeros((n, 11))
    imageUnLabeled[:, 0:2] = unlabeledWithDescription

    # Part 6: Create bounding boxes for our testing images
    for i in range(n):
        imgNum = int(imageUnLabeled[i, 0])
        imgi = images[:, :, :, imgNum]
        h, w, c = imgi.shape

        # show the image
        # pilimg = Image.fromarray(imgi, 'RGB')
        # pilimg.show()

        # bbox size [k,5] where n is image number, k is num of objects in each image
        # last dimension has x, y, height, width, depth of each bbox in image i
        bbox = find_BB_and_depth(imgi, depths[:, :, imgNum], False)

        # get the bouning box number
        k = int(imageUnLabeled[i, 1])

        # add the bbox values to the imageLabel
        imageUnLabeled[i, 4:9] = bbox[k]

        # add the height width of the image to the imageLabels
        imageUnLabeled[i, 9:11] = (h, w)

    Xtest_height = (imageUnLabeled[:, 10] / 2 -
                    imageUnLabeled[:, 7]) * imageUnLabeled[:, 8]
    Xtest_width = (imageUnLabeled[:, 10] / 2 -
                   imageUnLabeled[:, 7]) * imageUnLabeled[:, 8]

    # Old
    # Xtest_height = imageUnLabeled[:, [6, 8, 9]]
    # Xtest_width = imageUnLabeled[:, [7, 8, 10]]

    # Part 6: Fit a Linear Regression with training data
    #do training on linear regression
    linreg_x = LinearRegression(regLambda=1E-8)
    linreg_y = LinearRegression(regLambda=1E-8)

    linreg_x.fit(train_height, label_height)
    linreg_y.fit(train_width, label_width)

    # Part 7: predict heights and widths using linreg
    yHatHeight = linreg_x.predict(Xtest_height)
    yHatWidth = linreg_y.predict(Xtest_width)

    print "Linear Regression"
    print("yHatHeight:" + str(yHatHeight))
    print("yHatWidth:" + str(yHatWidth))

    # Part 8: Fit a Neural nets with training data

    #Get X train
    px = imageLabels[:, 9] / 2
    depth = imageLabels[:, 8]
    height = imageLabels[:, 6]
    py = imageLabels[:, 10] / 2
    width = imageLabels[:, 7]

    Xtrain_height = np.c_[height, px, depth]
    Xtrain_width = np.c_[width, py, depth]

    # Get y train
    #Ytrain = np.vstack((label_height, label_width)).T
    Ytrain_height = label_height.T
    Ytrain_width = label_width.T

    #Get x test
    px_test = imageUnLabeled[:, 9] / 2
    depth_test = imageUnLabeled[:, 8]
    height_test = imageUnLabeled[:, 6]
    py_test = imageUnLabeled[:, 10] / 2
    width_test = imageUnLabeled[:, 7]

    Xtest_height = np.c_[height_test, px_test, depth_test]
    Xtest_width = np.c_[width_test, py_test,depth_test]

    #get the predicted height/width
    y_hat_height = runDNN(Xtrain_height, Ytrain_height, Xtest_height)
    y_hat_width = runDNN(Xtrain_width, Ytrain_width, Xtest_width)

    print "Deep Neural Net"
    print("yHatHeight:" + str(y_hat_height))
    print("yHatWidth:" + str(y_hat_width))


if __name__ == '__main__':
    estimateSize()
