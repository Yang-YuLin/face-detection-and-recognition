# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 21:24:03 2018

@author: 杨玉林
"""

import cv2

cv2.namedWindow('catch video stream')

cap=cv2.VideoCapture(0)#读取视频

classifier=cv2.CascadeClassifier("D:\\jiance\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_alt2.xml")#使用人脸分类器

color=(0,255,0)

num=1 
catch_pic_num=1300
 
while(cap.isOpened()):
    ok,frame=cap.read()#读取一帧视频
    if not ok:#检验第一帧是否为空
        break

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faceRects=classifier.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=3,minSize=(32,32))
    if len(faceRects)>0:#大于0则检测到人脸 
        for faceRect in faceRects:#框出每一张人脸
            x,y,w,h=faceRect
            img_name='%s/%d.jpg'%("D:\\face detection\\data\\renlian34",num)#将当前帧保存为图片
            image=frame[y-10:y+h+10,x-10:x+w+10]
            cv2.imwrite(img_name,image)#完成保存的工作

            num+=1
            if num>(catch_pic_num):
                break

            cv2.rectangle(frame,(x-10,y-10),(x+w+10,y+h+10),color,2)

            font=cv2.FONT_HERSHEY_SIMPLEX#使用默认字体
            cv2.putText(frame,'num:%d'%(num),(x+30,y+30),font,1,(255,0,255),4)

    if num>(catch_pic_num):
            break

    cv2.imshow('catch video stream',frame)
    c=cv2.waitKey(10)
    if c & 0xFF==ord('q'):
        break

#释放摄像头并销毁所有窗口       
cap.release()
cv2.destroyAllWindows()