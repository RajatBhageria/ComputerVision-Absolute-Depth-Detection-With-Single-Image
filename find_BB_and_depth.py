import numpy as np
import cv2

# takes in a rgbd 4 dimensional image
def find_BB_and_depth(img_rgb, pixel_depths, drawContours = False):
    img = cv2.pyrDown(img_rgb)

    ret, threshed_img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)
    image, contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    output = np.zeros([len(contours), 5])

    for i in range(len(contours)):
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(contours[i])

        # draw a green rectangle to visualize the bounding rect
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        depth = np.mean(pixel_depths[2 * y:2 * (y + h), 2 * x:2 * (x + w)])
        if w > 15 and h > 15 and w < 600 and h < 400:
            output[i, :] = x * 2, y * 2, w * 2, h * 2, depth

<<<<<<< Updated upstream
    if drawContours :
        cv2.drawContours(img, contours, -1, (255, 255, 0), 1)
        cv2.imshow("contours", img)
        cv2.waitKey(0)
=======
    cv2.drawContours(img, contours, -1, (255, 255, 0), 1)
    cv2.imshow("contours", img)
    cv2.waitKey(0)
>>>>>>> Stashed changes
    cv2.destroyAllWindows()

    return output
    # the type is a [k, 5] array, the 5 features are x, y, w, h, depth_of_bb_in_meters
