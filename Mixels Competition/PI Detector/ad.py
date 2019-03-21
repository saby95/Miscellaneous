import cv2
import numpy as np

im = cv2.imread('circ.png')

output = im.copy()
flag = 0
image = im.copy()

gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5,5), 0)
#gray = cv2.bilateralFilter(gray,9,75,75)
maxValue = 255
adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C#cv2.ADAPTIVE_THRESH_MEAN_C #cv2.ADAPTIVE_THRESH_GAUSSIAN_C
thresholdType = cv2.THRESH_BINARY#cv2.THRESH_BINARY #cv2.THRESH_BINARY_INV
blockSize = 5 #odd number like 3,5,7,9,11
C = -3 # constant to be subtracted
thresh1 = cv2.adaptiveThreshold(gray, maxValue, adaptiveMethod, thresholdType, blockSize, C) 
#ret, thresh2 = cv2.threshold(thresh1,127,255,cv2.THRESH_BINARY_INV)
thresh1 = cv2.GaussianBlur(thresh1, (3,3), 5)

cv2.imshow('in',im)
cv2.imshow('out',thresh1)
#cv2.imshow('out1',thresh2)
cv2.waitKey(0)
cv2.destroyAllWindows()

(cnts, _) = cv2.findContours(thresh1.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)

for c in cnts:
    cv2.drawContours(image,c,-1,(0,255,0),2)
    flag = flag + 1

print flag
cv2.imshow('contours',image)
cv2.waitKey(0)
cv2.destroyAllWindows()

##circles = cv2.HoughCircles(im_thresholded, cv2.cv.CV_HOUGH_GRADIENT, 1.4,50,param1=40,param2=56,minRadius=0,maxRadius=0)
## 
### ensure at least some circles were found
##if circles is not None:
##	# convert the (x, y) coordinates and radius of the circles to integers
##	circles = np.round(circles[0, :]).astype("int")
## 
##	# loop over the (x, y) coordinates and radius of the circles
##	for (x, y, r) in circles:
##		# draw the circle in the output image, then draw a rectangle
##		# corresponding to the center of the circle
##		cv2.circle(output, (x, y), r, (0, 255, 0), 4)
##		cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
##		flag = flag + 1
## 
##	# show the output image
##	cv2.imshow("input",image)
##	cv2.imshow("output", output)
##	print flag
##	cv2.waitKey(0)
##	cv2.destroyAllWindows()
