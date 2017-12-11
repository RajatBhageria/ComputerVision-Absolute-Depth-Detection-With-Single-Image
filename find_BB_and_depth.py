import numpy as np
import cv2

def (img_d):
	img_no_d = img_d[:,:,:,]
	img = cv2.pyrDown(img_no_d)
	
	ret, threshed_img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)
	image, contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	
	output = np.zeros([len(contours),5])
	
	for i in range(len(contours)):
		# get the bounding rect
		x, y, w, h = cv2.boundingRect(contours[i])
		
		# draw a green rectangle to visualize the bounding rect
		cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
		output[i,:] = [x,y,w,h,depth]
	
	#cv2.drawContours(img, contours, -1, (255, 255, 0), 1)
	#cv2.imshow("contours", img)
	cv2.destroyAllWindows()
	
	output = x, y, p_width, p_height, depth
	return output