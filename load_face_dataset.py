# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 21:26:11 2018

@author: 杨玉林
"""

import cv2  
import os  
import numpy as np 

IMAGE_SIZE=64

def resize_image(image,height=IMAGE_SIZE,width=IMAGE_SIZE):  
    top,bottom,left,right=(0,0,0,0)  
      
    h,w,_=image.shape  
      
    longest_edge=max(h,w)  
      
    if h<longest_edge:  
        dh=longest_edge-h  
        top=dh//2  
        bottom=dh-top  
    elif w<longest_edge:  
        dw=longest_edge-w  
        left=dw//2  
        right=dw-left  
    else:  
        pass  
      
    BLACK=[0,0,0]  
  
    constant=cv2.copyMakeBorder(image,top,bottom,left,right,cv2.BORDER_CONSTANT,value=BLACK)#给图像增加边界，使图片长、宽等长  
      
    return cv2.resize(constant,(height,width))  
    
images=[]  
labels=[]  

def read_path(path_name):  
    for dir_item in os.listdir(path_name):  
        full_path=os.path.abspath(os.path.join(path_name,dir_item))  
          
        if os.path.isdir(full_path):#文件夹，继续递归调用  
            read_path(full_path)  
        else:#文件  
            if dir_item.endswith('.jpg'):  
                image=cv2.imread(full_path)  
                image=resize_image(image,IMAGE_SIZE,IMAGE_SIZE)  
                  
                images.append(image)  
                labels.append(path_name)  
    return images,labels  
    
def load_dataset(path_name):  
    images,labels=read_path(path_name)  
      
    images=np.array(images)#将输入的所有图片转成四维数组，尺寸为(图片数量*IMAGE_SIZE*IMAGE_SIZE*3)，图片为64 * 64像素,一个像素3个颜色值(RGB)  
    print(images.shape)  
      
      
    labels=np.array([1 if label.endswith('renlian1')   
                     else 2 if label.endswith('renlian2')    
                            else 3 if label.endswith('renlian3')    
                                   else 4 if label.endswith('renlian4')    
                                          else 5 if label.endswith('renlian5')   
                                                 else 6 if label.endswith('renlian6')    
                                                        else 7 if label.endswith('renlian7')   
                                                               else 8 if label.endswith('renlian8')    
                                                                      else 9 if label.endswith('renlian9')    
                                                                             else 10 if label.endswith('renlian10')  
                                                                                     else 11 if label.endswith('renlian11')  
                                                                                             else 12 if label.endswith('renlian12')  
                                                                                                     else 13 if label.endswith('renlian13')  
                                                                                                             else 14 if label.endswith('renlian14')  
                                                                                                                     else 15 if label.endswith('renlian15')  
                                                                                                                             else 16 if label.endswith('renlian16')  
                                                                                                                                     else 17 if label.endswith('renlian17')  
                                                                                                                                             else 18 if label.endswith('renlian18')  
                                                                                                                                                     else 19 if label.endswith('renlian19')  
                                                                                                                                                             else 20 if label.endswith('renlian20')  
                                                                                                                                                                     else 21 if label.endswith('renlian21')  
                                                                                                                                                                             else 22 if label.endswith('renlian22')  
                                                                                                                                                                                     else 23 if label.endswith('renlian23')  
                                                                                                                                                                                             else 24 if label.endswith('renlian24')  
                                                                                                                                                                                                     else 25 if label.endswith('renlian25')  
                                                                                                                                                                                                             else 26 if label.endswith('renlian26')  
                                                                                                                                                                                                                     else 27 if label.endswith('renlian27')  
                                                                                                                                                                                                                             else 28 if label.endswith('renlian28')  
                                                                                                                                                                                                                                     else 29 if label.endswith('renlian29')  
                                                                                                                                                                                                                                             else 30 if label.endswith('renlian30')  
                                                                                                                                                                                                                                                     else 31 if label.endswith('renlian31')  
                                                                                                                                                                                                                                                             else 32 if label.endswith('renlian32')  
                                                                                                                                                                                                                                                                     else 33 if label.endswith('renlian33')  
                                                                                                                                                                                                                                                                             else 34 for label in labels])  
    return images,labels 

if __name__ == '__main__':  
     images, labels = load_dataset('D:\\face detection\\data')                             