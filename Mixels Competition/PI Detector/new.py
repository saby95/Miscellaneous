# import the necessary packages
import numpy as np
import argparse
import cv2
flag = 0

# load the image, clone it for output, and then convert it to grayscale
image = cv2.imread('circ2.png')
height,width,channels = image.shape
print width,height,channels
if width>1000 :
        height = height/2
        width = width/2
        image = cv2.resize(image,(width,height))
output = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray,(9,9),1)
gray = cv2.Canny(gray, 50, 200)
gray = cv2.bilateralFilter(gray,9,75,75)
#output = gray.copy()
cv2.imshow("output", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()


#gray = cv2.bilateralFilter(gray,9,75,75)

# detect circles in the image
circles = cv2.HoughCircles(gray, cv2.cv.CV_HOUGH_GRADIENT, 1.5,50,param1=40,param2=56,minRadius=0,maxRadius=0)
 
# ensure at least some circles were found
if circles is not None:
	# convert the (x, y) coordinates and radius of the circles to integers
	circles = np.round(circles[0, :]).astype("int")
 
	# loop over the (x, y) coordinates and radius of the circles
	for (x, y, r) in circles:
		# draw the circle in the output image, then draw a rectangle
		# corresponding to the center of the circle
		cv2.circle(output, (x, y), r, (0, 255, 0), 4)
		cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
		flag = flag + 1
 
	# show the output image
	cv2.imshow("input",image)
	cv2.imshow("output", output)
	print flag
	cv2.waitKey(0)
	cv2.destroyAllWindows()
