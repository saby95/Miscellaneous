import numpy as np
import sys
import cv2

ab = [0] * 20
j = 20

# load the image
image = cv2.imread(str(sys.argv[1]))
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray,(5,5),0)
gray = cv2.bilateralFilter(gray,9,75,75)
##cv2.imshow("org", gray)

edged = cv2.Canny(gray,75,300)
##cv2.imshow("edge",edged)

# find the contours in the mask
(cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)

# loop over the contours
for c in cnts:
        # draw the contour and show it
        area = cv2.contourArea(c)
        if area < 10:
                continue
        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        peri = cv2.arcLength(c,True)
        approx = cv2.approxPolyDP(c,0.02*peri,True)
        h = len(approx)
        ab[h] = ab[h]+1

#print the shapes and their numbers
for k in range(20) :
        if ab[k] == 0:
             continue
        print "%d - %d" %(k,ab[k])
cv2.destroyAllWindows()

#Show the detected shapes
#cv2.imshow("Image", image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
