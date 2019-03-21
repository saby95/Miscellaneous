import cv2                              
import numpy as np
import math
import argparse
import sys

def order_points(pts):
        # initialzie a list of coordinates that will be ordered
        # such that the first entry in the list is the top-left,
        # the second entry is the top-right, the third is the
        # bottom-right, and the fourth is the bottom-left
        rect = np.zeros((4, 2), dtype = "float32")

        # the top-left point will have the smallest sum, whereas
        # the bottom-right point will have the largest sum
        s = pts.sum(axis = 1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # now, compute the difference between the points, the
        # top-right point will have the smallest difference,
        # whereas the bottom-left will have the largest difference
        diff = np.diff(pts, axis = 1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        # return the ordered coordinates
        return rect

def four_point_transform(image, pts):
        # obtain a consistent order of the points and unpack them
        # individually
        rect = order_points(pts)
        (tl, tr, br, bl) = rect

        # compute the width of the new image, which will be the
        # maximum distance between bottom-right and bottom-left
        # x-coordiates or the top-right and top-left x-coordinates
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        # compute the height of the new image, which will be the
        # maximum distance between the top-right and bottom-right
        # y-coordinates or the top-left and bottom-left y-coordinates
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image, again specifying points
        # in the top-left, top-right, bottom-right, and bottom-left
        # order
        dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]], dtype = "float32")

        # compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

        # return the warped image
        return warped

##ap = argparse.ArgumentParser()
##ap.add_argument("-i", "--image", required = True,
##	help = "Path to the image to be scanned")
##args = vars(ap.parse_args())
##
##ap.add_argument("-i", "--oimage", required = True,
##	help = "Path to the image to be scanned")
##args1 = vars(ap.parse_args())

image = cv2.imread(str(sys.argv[1]))
ratio = image.shape[0] / 700.0
orig = image.copy()
image = cv2.resize(image, (int(round(image.shape[1]/ratio)),700))

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray,9,75,75)
ret1,thresh1 = cv2.threshold(gray,175,255,cv2.THRESH_BINARY)

print "STEP 1:  Thresholding"
##cv2.imshow('threshold',thresh1)
##cv2.waitKey(0)
##cv2.destroyAllWindows()

(cnts, _) = cv2.findContours(thresh1.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

dp = 0.02

length = 0
for c in cnts:
        peri = cv2.arcLength(c, True)
        if peri > length:
                length = peri

# loop over the contours
for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        if peri == length :
                approx = cv2.approxPolyDP(c, dp* peri, True)
                screenCnt = approx

                # if our approximated contour has four points, then we
                # can assume that we have found our screen
                for im in screenCnt:
                        cv2.circle(thresh1,(int(im[0][0]),int(im[0][1])),1,(255,0,0),10)

# show the contour (outline) of the piece of paper
print "STEP 2: Find contours of paper"
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
#cv2.imshow("Outline", image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# apply the four point transform to obtain a top-down
# view of the original image
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

# convert the warped image to grayscale, then threshold it
# to give it that 'black and white' paper effect
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
#warped1 = cv2.adaptiveThreshold(warped,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#                               cv2.THRESH_BINARY,11,2)
#warped = warped.astype("uint8") * 255
warped1 = cv2.adaptiveThreshold(warped.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3)

# show the original and scanned images
print "STEP 3: Apply perspective transform"
height1,width1,channels1 = orig.shape
height2,width2 = warped.shape
ratio1 = float(height1)/650
ratio2 = float(height2)/650
##cv2.imshow("Original", cv2.resize(orig,(int(round(width1/ratio1)),650)))
##cv2.imshow("Scanned", cv2.resize(warped,(int(round(width2/ratio2)),650)))
##cv2.imshow("Scanned and Thresholded", cv2.resize(warped1,(int(round(width2/ratio2)),650)))
cv2.imwrite(str(sys.argv[2]),warped1)

print "Done"
#cv2.imshow("Scanned", warped)
##cv2.waitKey(0)
##cv2.destroyAllWindows()


