face detection and recognition
==
face detection利用了opencv完成\
face recognition利用了CNN模型完成

预装工具 
--
opencv\
tensorflow\
keras（深度学习库）\
sklearn\
anaconda3 语言环境 python3.5

项目流程 
--
从电脑摄像头获取实时视频流\
利用opencv检测出人脸\
准备人脸数据\
对数据进行处理及加载\
利用Keras库训练人脸识别模型\
从电脑的摄像头中识别出我自己

文件说明
--
catch_video.py：视频截取及按帧读取，完成项目流程中的前三步工作\
load_face_dataset.py：完成项目流程中的第四步工作\
face_train_use_keras.py：完成项目流程中的第五步工作\
face_predict_use_keras.py：完成项目流程中的最后一步工作
