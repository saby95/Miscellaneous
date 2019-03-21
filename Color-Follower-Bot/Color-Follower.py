import serial
import numpy as np
import cv2
import math
import time


cap=cv2.VideoCapture(0)
ard=serial.Serial('COM24',9600)
flag=0
r=0
fl=0
vis=0
flg_full = 0
flg_g = flg_b = flg_r = 0
flag_fir = 1
flag_cal = 0
global frame
#print 'connected'


lg = np.array([39,54,75])
ug = np.array([103,196,160])
lb = np.array([111,74,56])
ub = np.array([191,191,170])
lr = np.array([0,162,78])
ur = np.array([10,222,160])
lbl = np.array([0,0,0])
ubl = np.array([255,255,57])

lf = np.array([106,126,131])
uf = np.array([255,253,243])
lbk = np.array([0,132,141])
ubk = np.array([80,221,182])

greeni = [lg,ug]
bluei = [lb,ub]
redi = [lr,ur]
blacki = [lbl,ubl]

fronti = [lf,uf]
backi = [lbk,ubk]

contoursf = contoursbk = contoursbl = contoursr = contoursg = contoursb = contoursd = 0
coin_t = coin_c = coin_b = 0
contourss = contourso = 0
xg=yg=xbu=ybu=xr=yr=xs=ys= 0
xo = yo =[]

def perspectivate(c):
    #c = contoursbl[2]
    global screenCnt
    peri = cv2.arcLength(c, True)
    #if peri == length :
    dp = 0.02
    approx = cv2.approxPolyDP(c, dp* peri, True)
    screenCnt = approx

    # if our approximated contour has four points, then we
    # can assume that we have found our screen
    j = 0
    if len(approx) != 4:
            temp = np.zeros((4,1,2))
            for k in range(len(screenCnt)):
                    for tp in temp:
                            if tp[0].all() == 0:
                                    continue
                            n1 = math.sqrt(((tp[0][0]-screenCnt[k][0][0])*(tp[0][0]-screenCnt[k][0][0]))+((tp[0][1]-screenCnt[k][0][1])*(tp[0][1]-screenCnt[k][0][1])))
                            if n1 < 100 :
                                     continue
                             
                            temp[j][0] = screenCnt[k][0]
                            j = j + 1
                    screenCnt = temp
    return screenCnt


def order_points(pts):

        rect = np.zeros((4, 2), dtype = "float32")

        s = pts.sum(axis = 1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]


        diff = np.diff(pts, axis = 1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        # return the ordered coordinates
        return rect

def four_point_transform(image, pts):

        rect = order_points(pts)
        (tl, tr, br, bl) = rect


        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))


        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))


        dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]], dtype = "float32")
        
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

        return warped


def perspec():
##    global frame
    global screenCnt
    while(True):
        ret,frame = cap.read()
        img = np.zeros((frame.shape[0],frame.shape[1],3))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray,9,75,75)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        #gray = cv2.bilateralFilter(gray,9,75,75)
        edged = cv2.Canny(gray, 30, 200)
        (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

        try:
            cv2.drawContours(img,cnts[0], -1, (255,0,0), 3)
        except Exception:
            continue

        cv2.imshow('img',img)
        if cv2.waitKey(1) & 0xFF == 27:
            screenCnt = perspectivate(cnts[0])
            return screenCnt

    cv2.destroyAllWindows()





def update_frame():
    global flg_full,flag,flg_g,flg_r,flg_b,contoursf,contoursbk,contoursbl,contourss
    global contoursg,contoursr,contoursb,coin_c,coin_t,coin_b, frame,contourso
    global screenCnt
    global xg,yg,xbu,ybu,xr,yr,xs,ys,xo,yo,contourss,contourso
    global flag_fir
    try:
        ret, frame=cap.read()
        img = np.zeros((frame.shape[0],frame.shape[1],3))

        frame = four_point_transform(frame, screenCnt.reshape(4, 2))
        frame = cv2.resize(frame, (640,480))
        frame = frame[9:469 , 9:629]
        frame =cv2.resize(frame, ( 640, 480))

        hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)


        front = cv2.inRange(hsv,fronti[0],fronti[1])
        back = cv2.inRange(hsv,backi[0],backi[1])
        green = cv2.inRange(hsv,greeni[0],greeni[1])
        black = cv2.inRange(hsv,blacki[0],blacki[1])
        blue = cv2.inRange(hsv,bluei[0],bluei[1])
        red = cv2.inRange(hsv,redi[0],redi[1])
        
        kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        
        front=cv2.erode(front,kernel)
        front=cv2.dilate(front,kernel)
        back=cv2.erode(back,kernel)
        back=cv2.dilate(back,kernel)
        
        
        green=cv2.erode(green,kernel)
        green=cv2.dilate(green,kernel)
        red = cv2.erode(red,kernel)
        red = cv2.dilate(red,kernel)
        blue =cv2.erode(blue,kernel)
        blue = cv2.dilate(blue,kernel)
        black = cv2.erode(black,kernel)
        black = cv2.dilate(black,kernel)
        
        
        contoursf,hierarchyf=cv2.findContours(front,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        contoursbk,hierarchyb=cv2.findContours(back,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        contoursbl,hierarchyd=cv2.findContours(black,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        contoursg,hierarchyd=cv2.findContours(green,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        contoursb,hierarchyd=cv2.findContours(blue,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        contoursr,hierarchyd=cv2.findContours(red,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        contoursbl = sorted(contoursbl, key = cv2.contourArea, reverse = True)[:4]
        
        if(flag_cal == 0):
            cv2.drawContours(img, contoursf, -1, (255,0,0), 3)
            cv2.drawContours(img, contoursbk, -1, (255,0,0), 3)
            cv2.drawContours(img, contoursbl, -1, (255,0,0), 3)
            cv2.drawContours(img, contoursg, -1, (255,0,0), 3)
            cv2.drawContours(img, contoursr, -1, (255,0,0), 3)
            cv2.drawContours(img, contoursb, -1, (255,0,0), 3)
            cv2.imshow('img',img)
            cv2.waitKey(1)

        
##        k = 0
##        j = 0
##        while(j == 0):
##            k = 0
##            for i in range(len(contoursbk)):
##                print i,len(contoursbk)
##                if(cv2.contourArea(contoursbk[i])<400):
##                    contoursbk = np.delete(contoursbk, i)
##                    k = 1
##                if(k == 1):
##                    break
##            if(i == (len(contoursbk)-1)):
##                break
##
##        k = 0
##        j = 0
##        while(j == 0):
##            k = 0
##            for i in range(len(contoursf)):
##                print i,len(contoursf)
##                if(cv2.contourArea(contoursf[i])<400):
##                    contoursf = np.delete(contoursf, i)
##                    k = 1
##                if(k == 1):
##                    break
##            if(i == (len(contoursf)-1)):
##                break
##
##        k = 0
##        j = 0
##        while(j == 0):
##            k = 0
##            for i in range(len(contoursg)):
##                print i,len(contoursg)
##                if(cv2.contourArea(contoursg[i])<500):
##                    contoursg = np.delete(contoursg, i)
##                    k = 1
##                if(k == 1):
##                    break
##            if(i == (len(contoursg)-1)):
##                break
##
##        k = 0
##        j = 0
##        while(j == 0):
##            k = 0
##            for i in range(len(contoursb)):
##                print i,len(contoursb)
##                print 'a'
##                if(cv2.contourArea(contoursb[i])<500):
##                    contoursg = np.delete(contoursb, i)
##                    k = 1
##                if(k == 1):
##                    break
##            if(i == (len(contoursb)-1)):
##                break
##
##        k = 0
##        j = 0
##        while(j == 0):
##            k = 0
##            for i in range(len(contoursr)):
##                print i,len(contoursr)
##                print 'b'
##                if(cv2.contourArea(contoursr[i])<500):
##                    contoursg = np.delete(contoursr, i)
##                    k = 1
##                if(k == 1):
##                    break
##            if(i == (len(contoursr)-1)):
##                break

        if(flag_fir ==0):
                
            contoursg = sorted(contoursg, key = cv2.contourArea, reverse = True)[:1]
            contoursb = sorted(contoursb, key = cv2.contourArea, reverse = True)[:1]
            contoursr = sorted(contoursr, key = cv2.contourArea, reverse = True)[:1]
            M1 = cv2.moments(contoursg[0])
            M2 = cv2.moments(contoursb[0])
            M3 = cv2.moments(contoursr[0])
            M4 = cv2.moments(contoursbl[0])
            contourss = contoursbl[0]
            xg = int(M1['m10']/M1['m00'])
            yg = int(M1['m01']/M1['m00'])
            xbu = int(M2['m10']/M2['m00'])
            ybu = int(M2['m01']/M2['m00'])
            xr = int(M3['m10']/M3['m00'])
            yr = int(M3['m01']/M3['m00'])
            xs = int(M4['m10']/M4['m00'])
            ys = int(M4['m01']/M4['m00'])
            contourso = []
            for i in range(len(contoursbl)):
                if i == 0:
                    continue
                M4 = cv2.moments(contoursbl[i])
                xo.append(int(M4['m10']/M4['m00']))
                yo.append(int(M4['m01']/M4['m00']))
                contourso.append(contoursbl[i])
            flag_fir = 1

    except Exception:
        print Exception
        return
        




def reroute(xtd,ytd,num):
        
        while(True):
            update_frame()
            img = np.zeros((frame.shape[0],frame.shape[1],3))

            if(num == 0):
                contoursd = contoursg
            elif(num == 1):
                contoursd = contoursb
            elif(num == 2):
                contoursd = contoursr
            if (num !=3):
                if ((len(contoursd) == 0)):
                    ard.write('3')
                    break
##                            elif(len(contoursd) == 1):
##                                if(cv2.contourArea(contoursd[0]) < 50):
##                                    ard.write('3')
##                                    break
            
                    
            cv2.drawContours(img, contoursf, -1, (255,0,0), 3)
            cv2.drawContours(img, contoursbk, -1, (255,0,0), 3)
            #cv2.drawContours(img, contoursd, -1, (255,0,0), 3)
            
            cv2.circle(img,(xtd,ytd),1,(0,0,255),2)
            cv2.circle(img,(xd,yd),1,(0,255,0),2)
            
            M1=cv2.moments(contoursf[0])
            M2=cv2.moments(contoursbk[0])

            xf=int(M1['m10']/M1['m00'])
            yf=int(M1['m01']/M1['m00'])
            cv2.circle(img,(xf,yf),1,(0,0,255),2)

            xb=int(M2['m10']/M2['m00'])
            yb=int(M2['m01']/M2['m00'])
            cv2.circle(img,(xb,yb),1,(0,0,255),2)


            x=(xf+xb)/2
            y=(yf+yb)/2

            cv2.circle(img,(x,y),1,(0,0,255),2)

            dst=math.sqrt(((xtd-xf)*(xtd-xf))+((ytd-yf)*(ytd-yf)))

            #print dst

            thd=(math.atan2(ytd-y,xtd-x)*180)/math.pi
            thb=(math.atan2(yf-yb,xf-xb)*180)/math.pi
            if thd<0:
                thd=thd+360
            if thb<0:
                thb=thb+360
                
            phi=thd-thb
            if phi>180:
                phi=phi-360
            if phi<-180:
                phi=phi+360
                
            #print phi

            if phi>15:
                ard.write('0')
            if phi<-15:
                ard.write('1')
            if((phi<15) & (phi>-15)):
                ard.write('2')
            if dst<=10:
                ard.write('0')
                break

            cv2.imshow('frame',img)
            if cv2.waitKey(1)==27:
                ard.write('3')
                break


def track(num) :
    global flg_full,flag,flg_g,flg_r,flg_b,contoursf,contoursbk,contoursbl
    global contoursg,contoursr,contoursr,coin_c,coin_t,coin_b, frame,xd,yd
    global xg,yg,xbu,ybu,xr,yr,xs,ys,xo,yo,contourss,contourso
    fg = 0
    xd=yd=0
    while(True):
        update_frame()
        img = np.zeros((frame.shape[0],frame.shape[1],3))
        
        if(num == 0):
            contoursd = contoursg
            [xd,yd] = [xg,yg]
            cv2.drawContours(img, contoursg, -1, (54,255,0), 3)
        elif(num == 1):
            contoursd = contoursb
            [xd,yd] = [xbu,ybu]
            cv2.drawContours(img, contoursb, -1, (54,255,0), 3)
        elif(num == 2):
            contoursd = contoursr
            [xd,yd] = [xr,yr]
            cv2.drawContours(img, contoursr, -1, (54,255,0), 3)
        
        contoursd = sorted(contoursd, key = cv2.contourArea, reverse = True)   
        if ((len(contoursd) == 0)):
            ard.write('3')
            break

            
##        elif(len(contoursd) == 1):
##            if(cv2.contourArea(contoursd[0]) < 50):
##                ard.write('3')
##                break





        contoursf = sorted(contoursf, key = cv2.contourArea, reverse = True)
        contoursbk = sorted(contoursbk, key = cv2.contourArea, reverse = True)
        cv2.drawContours(img, contoursf, -1, (255,54,0), 3)
        cv2.drawContours(img, contoursbk, -1, (255,0,54), 3)
        cv2.drawContours(img, contoursbl, -1, (78,78,54), 3)
        
        print 'running'        
          #if ((len(contoursd)>0) | (fg==1)):
        M1=cv2.moments(contoursf[0])
        M2=cv2.moments(contoursbk[0])
        #M3=cv2.moments(contoursd[0])

        xf=int(M1['m10']/M1['m00'])
        yf=int(M1['m01']/M1['m00'])
        cv2.circle(img,(xf,yf),1,(0,255,0),2)

        xb=int(M2['m10']/M2['m00'])
        yb=int(M2['m01']/M2['m00'])
        cv2.circle(img,(xb,yb),1,(0,255,0),2)

        cv2.circle(img,(xd,yd),1,(0,255,0),2)
        x=(xf+xb)/2
        y=(yf+yb)/2
        cv2.circle(img,(x,y),1,(0,0,255),2)
       
        dst=math.sqrt(((xd-x)*(xd-x))+((yd-y)*(yd-y)))

        #print dst

        thd=(math.atan2(yd-y,xd-x)*180)/math.pi
        thb=(math.atan2(yf-yb,xf-xb)*180)/math.pi
        phi_o = []

        if thd<0:
            thd=thd+360
        if thb<0:
            thb=thb+360
            
        phi=thd-thb
        
        for i in range(len(xo)):
            temp = (math.atan2(yo[i]-y,xo[i]-x)*180)/math.pi
            if temp < 0:
                temp = temp + 360
            phi_o.append(temp-thb)

            if phi_o[i] > 180 :
                phi_o[i] = phi_o[i] - 360
            if phi_o[i] < -180 :
                phi_o[i] = phi_o[i] + 360

        if phi>180:
            phi=phi-360
        if phi<-180:
            phi=phi+360
            
        #print phi

        if phi>15:
            ard.write('0')
        if phi<-15:
            ard.write('1')
        print phi    
        if phi<15:
          if phi>-15:
##                for i in range(len(phi_o)):
##                    if ((phi_o[i]<15)&(phi_o[i]>-15)):
##
##                        thd1=(math.atan2(yd-yo[i],xd-xo[i])*180)/math.pi
##                        thd2=(math.atan2(yo[i]-y,xo[i]-x)*180)/math.pi
##                        
##                        if thd1<0:
##                            thd1=thd1+360
##                        if thd2<0:
##                            thd2=thd2+360
##                        phi_d = thd1
####                        if phi_d>180:
####                            phi_d = phi_d-360
####                        if phi_d<-180:
####                            phi_d = phi_d+360
##                        
##                        xtd = []
##                        ytd = []
##                        for i in range(-60,60,30):
##                                
##                            if(phi_o>0):
##                                phi_d1 = (phi_d-90+i)
##                            else:
##                                phi_d1 = (phi_d+90-i)
##
##                            leng = 70
##                            xttd = int(round(xo[i]+leng*math.cos(math.radians(phi_d1))))
##                            yttd = int(round(yo[i]+leng*math.sin(math.radians(phi_d1))))
##                            xtd.append(xttd)
##                            ytd.append(yttd)
##                            leng = leng + 5
##
##                        f=0
##                        for i in range(len(xtd)):
##                            reroute(xtd[i],ytd[i],num)
##                        

                ard.write('2')
        if dst<=10:
            ard.write('3')
            time.sleep(5)
            if num == 0:
                flg_g = 1
                coin_t = coin_t-1000
                coin_b = coin_b+1000
                #coin_c = coin_c+1000
            elif num == 1:
                flg_b = 1
                coin_t = coin_t-500
                coin_b = coin_b+500
                #coin_c = coin_c+500
            elif num == 2:
                flg_r = 1
                coin_t = coin_t-250
                coin_b = coin_b+250
                #coin_c = coin_c+250
            flg_full = flg_full+1
            break
        
        cv2.imshow('frame',img)
        if cv2.waitKey(1)==27:
            ard.write('3')
            break






def back_to_safe():
    global flg_full,flag,flg_g,flg_r,flg_b,contoursf,contoursbk,contoursbl
    global contoursg,contoursr,contoursr,coin_c,coin_t,coin_b, frame
    global xg,yg,xbu,ybu,xr,yr,xs,ys,xo,yo,contourss,contourso
    [xd,yd] = [xs,ys]
    while(True):
        
        update_frame()
        img = np.zeros((frame.shape[0],frame.shape[1],3), np.unit8)
            
        
        contoursf = sorted(contoursf, key = cv2.contourArea, reverse = True)
        contoursbk = sorted(contoursbk, key = cv2.contourArea, reverse = True)

        cv2.drawContours(img, contoursf, -1, (255,54,0), 3)
        cv2.drawContours(img, contoursbk, -1, (255,0,54), 3)
        cv2.drawContours(img, contoursbl, -1, (78,78,54), 3)


        #if len(contourss) == 1:
        M1=cv2.moments(contoursf[0])
        M2=cv2.moments(contoursbk[0])
        M3=cv2.moments(contourss[0])

        
        xf=int(M1['m10']/M1['m00'])
        yf=int(M1['m01']/M1['m00'])
        cv2.circle(img,(xf,yf),1,(0,255,0),2)
      
        xb=int(M2['m10']/M2['m00'])
        yb=int(M2['m01']/M2['m00'])
        cv2.circle(img,(xb,yb),1,(0,255,0),2)
        cv2.circle(img,(xd,yd),1,(0,255,0),2)
      
        
        x=(xf+xb)/2
        y=(yf+yb)/2
        
        cv2.circle(img,(x,y),1,(0,0,255),2)
        
        dst=math.sqrt(((xd-x)*(xd-x))+((yd-y)*(yd-y)))
        
        #print dst
        
        thd=(math.atan2(yd-y,xd-x)*180)/math.pi
        thb=(math.atan2(yf-yb,xf-xb)*180)/math.pi
        phi_o = []
        
        if thd<0:
            thd=thd+360
        if thb<0:
            thb=thb+360
            
        phi=thd-thb
        
        for i in range(len(xo)):
            temp = (math.atan2(yo[i]-y,xo[i]-x)*180)/math.pi
            if temp < 0:
                temp = temp + 360
            phi_o.append(temp-thb)

            if phi_o[i] > 180 :
                phi_o[i] = phi_o[i] - 360
            if phi_o[i] < -180 :
                phi_o[i] = phi_o[i] + 360
        
        if phi>180:
            phi=phi-360
        if phi<-180:
            phi=phi+360

        if phi>15:
            ard.write('0')
        if phi<-15:
            ard.write('1')
        if phi<15:
          if phi>-15:
##                for i in range(len(phi_o)):
##                    if ((phi_o[i]<15)&(phi_o[i]>-15)):
##                        
##                        thd1=(math.atan2(yd-yo[i],xd-xo[i])*180)/math.pi
##                        thd2=(math.atan2(yo[i]-y,xo[i]-x)*180)/math.pi
##                        
##                        if thd1<0:
##                            thd1=thd1+360
##                        if thd2<0:
##                            thd2=thd2+360
##                        phi_d = thd1
####                        if phi_d>180:
####                            phi_d = phi_d-360
####                        if phi_d<-180:
####                            phi_d = phi_d+360
##                        
##                        xtd = []
##                        ytd = []
##                        for i in range(-60,60,30):
##                                
##                            if(phi_o>0):
##                                phi_d1 = (phi_d-90+i)
##                            else:
##                                phi_d1 = (phi_d+90-i)
##
##                            leng = 70
##                            xttd = int(round(xo[i]+leng*math.cos(math.radians(phi_d1))))
##                            yttd = int(round(yo[i]+leng*math.sin(math.radians(phi_d1))))
##                            xtd.append(xttd)
##                            ytd.append(yttd)
##                            leng =leng+5
##
##                        f=0
##                        for i in range(len(xtd)):
##                            reroute(xtd[i],ytd[i],3)
##                        

                                            
                ard.write('2')
        if dst<=10:
            ard.write('3')
            time.sleep(5)
            flg_r = flg_g = flg_b = flg_full = coin_b = 0
            coin_c = coin_c+coin_b
            break
        cv2.imshow('frame',img)
        if cv2.waitKey(1)==27:
            ard.write('3')
            break





screenCnt = perspec()
while(1):
    update_frame()
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == 27:
        cv2.destroyAllWindows()
        flag_fir =0
        break
while(1):
    update_frame()
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == 27:
        cv2.destroyAllWindows()
        flag_cal =0
        break
t = raw_input('Enter the coins to collect : ')
coins = int(t)
coin_t = coins
while(coin_t > 0):

    update_frame()
    
    if((flg_full == 1) | (coin_t - coin_b == 0)):
        back_to_safe()
    
    if((((coin_t - 1000) % 250) == 0) & (flg_g == 0) & ((coin_t - 1000 ) >= 0)):
        if(len(contoursg) >0):
            track(0)

    if((flg_full == 1) | (coin_t - coin_b == 0)):
        back_to_safe()

    update_frame()

    if((((coin_t - 500) % 250) == 0) & (flg_b == 0) & ((coin_t - 500 ) >= 0)):
        if((((coin_t - 1000) % 250) == 0) & (len(contoursg) == 1) & (flg_g == 0) & ((coin_t - 1000 ) >= 0)):
                track(0)
        else:
            if(len(contoursb) == 1):
                track(1)
    if((flg_full == 1) | (coin_t - coin_b == 0)):
        back_to_safe()

    update_frame()
        
    if((((coin_t - 250) % 250) == 0) & (flg_r == 0) & ((coin_t - 250 ) > 0)):
        if((((coin_t - 1000) % 250) == 0) & (len(contoursg) == 1) & (flg_g == 0) & ((coin_t - 1000 ) >= 0)):
                track(0)
        if((((coin_t - 500) % 250) == 0) & (len(contoursb) == 1) & (flg_b == 0) & ((coin_t - 500 ) >= 0)):
                track(1)
        else:
                track(2)
           

                
        
    
cap.release()
ard.close()
cv2.destroyAllWindows()
