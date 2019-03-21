import cv2
import numpy as np
import cv2.cv as cv
imga = cv2.imread('Q2_5.png',0)
img = cv2.medianBlur(imga,5)
#img=cv2.bilateralFilter(img,9,75,75)
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
circles = cv2.HoughCircles(img,cv.CV_HOUGH_GRADIENT,1.2,100,
param1=50,param2=20,minRadius=0,maxRadius=0)
circles = np.uint16(np.around(circles))
for i in circles[0,:]:
# draw the outer circle
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

cv2.imshow('detected circles',cimg)
cv2.waitKey(0)
cv2.destroyAllWindows()
