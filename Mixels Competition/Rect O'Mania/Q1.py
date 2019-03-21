import numpy as np
import cv2
import math
import sys

flag = 0

#img = cv2.imread(str(sys.argv[1]))
img = cv2.imread('rect.png')
imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret1,thresh1 = cv2.threshold(imgray,127,255,cv2.THRESH_BINARY)
ret2,thresh2 = cv2.threshold(imgray,127,255,cv2.THRESH_BINARY_INV)
contoursb1, hierarchyb1 = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contoursb2, hierarchyb2 = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

contoursb = contoursb1
if len(contoursb2)>len(contoursb1):
    contoursb = contoursb2

contoursb = sorted(contoursb, key = cv2.contourArea, reverse = True)
print len(contoursb)
    
for i in range(len(contoursb)):
    cntb=contoursb[i]            
    M=cv2.moments(cntb) #center of bot
    
    if M['m00']!=0 :            
        xb=int(M['m10']/M['m00'])
        yb=int(M['m01']/M['m00'])
        centrb=(xb,yb)
        cv2.circle(img,(xb,yb),1,(255,0,0),2)
        area = cv2.contourArea(cntb)
        rect = cv2.minAreaRect(cntb)
        box = cv2.cv.BoxPoints(rect)
        p1 = box[0]
        p2 = box[1]
        p3 = box[3]
        n1 = math.sqrt(((p1[0]-p2[0])*(p1[0]-p2[0]))+((p1[1]-p2[1])*(p1[1]-p2[1])))
        n2 = math.sqrt(((p1[0]-p3[0])*(p1[0]-p3[0]))+((p1[1]-p3[1])*(p1[1]-p3[1])))

        if n1<n2:
            x1 = int((p1[0]+p2[0])/2)
            y1 = int((p1[1]+p2[1])/2)
        if n2<n1:
            x1 = int((p1[0]+p3[0])/2)
            y1 = int((p1[1]+p3[1])/2)
        cv2.circle(img,(x1,y1),1,(0,0,255),7)

        if x1 != xb:
            angle=int((math.atan2(yb-y1,xb-x1)*180)/math.pi)
            if angle>=180 :
                angle = angle-180
            if angle<0 :
                angle = -(angle)
        if x1 == xb:
            angle = 90
        perimeter = cv2.arcLength(cntb,True)
        cv2.drawContours(img, contoursb, -1, (255,0,0), 3)
        
        print xb,yb,angle

#cv2.imshow('img',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

