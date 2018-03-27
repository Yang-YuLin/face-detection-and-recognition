# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 21:34:23 2018

@author: 杨玉林
"""

import cv2  
  
from face_train_use_keras import Model  
  
if __name__=='__main__':  
    model=Model()  
    model.load_model(file_path='D:\\face detection\\data\\me.face.model.h5')  
      
    color=(0,255,0)  
      
    cap=cv2.VideoCapture(0)  
      
    cascade_path='D:\\jiance\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_alt2.xml'  
      
    while True:  
        _,frame=cap.read()  
          
        frame_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)  
          
        cascade=cv2.CascadeClassifier(cascade_path)  
          
        faceRects=cascade.detectMultiScale(frame_gray,scaleFactor=1.2,minNeighbors=3,minSize=(32,32))  
        if len(faceRects)>0:  
            for faceRect in faceRects:  
                x,y,w,h=faceRect  
                cv2.rectangle(frame,(x-10,y-10),(x+w+10,y+h+10),color,thickness=2)  
                image=frame[y-10:y+h+10,x-10:x+w+10]  
                faceID=model.face_predict(image)  
                  
                if faceID==23:  
                    cv2.rectangle(frame,(x-10,y-10),(x+w+10,y+h+10),color,thickness=2)  
                    cv2.putText(frame,'yulin',(x+30,y+30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)  
                else:  
                    pass  
                      
        cv2.imshow('shi bie',frame)  
          
        k=cv2.waitKey(10)  
          
        if k & 0xFF==ord('q'):  
            break  
          
    cap.release()  
    cv2.destroyAllWindows()  
