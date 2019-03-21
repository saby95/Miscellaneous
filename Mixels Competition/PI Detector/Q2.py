# import the necessary packages
import numpy as np
import argparse
import sys
import cv2
flag = 0

# load the image, clone it for output, and then convert it to grayscale
#image = cv2.imread('circ2.png')
image = cv2.imread(str(sys.argv[1]))
height,width,channels = image.shape
if width>1000 :
        height = height/2
        width = width/2
        image = cv2.resize(image,(width,height))
output = image.copy()
src = image.copy()
flipped = cv2.flip( src, 1 )
flipped = np.concatenate( (src, flipped), axis = 1 )
src = np.concatenate( (flipped, src), axis = 1 )

flipped = cv2.flip( src, 0 )
flipped = np.concatenate( (src, flipped), axis = 0 )
src = np.concatenate( (flipped, src), axis = 0 )

src = cv2.flip( src, -1 )
output = src.copy()
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray,(9,9),2)
#gray = cv2.bilateralFilter(gray,9,75,75)

# detect circles in the image
circles = cv2.HoughCircles(gray, cv2.cv.CV_HOUGH_GRADIENT, 1,50,param1=40,param2=56,minRadius=0,maxRadius=0)

height = output.shape[0]
width = output.shape[1]
yx = height /3
xx = width / 3
wx = (width / 3)
hx = (height /3)

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
                if ((x > xx) & (x < (wx+xx))) & ((y > yx) & ( y < (hx+yx))):
                        flag = flag + 1
 
        # show the output image
        #cv2.imshow("input",image)
        #output1 = output[yx: yx + hx, xx: xx + wx]
        #cv2.imshow("output", output1)
        print flag
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
