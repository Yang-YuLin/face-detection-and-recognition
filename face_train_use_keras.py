# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 21:28:57 2018

@author: 杨玉林
"""

class Dataset:  
    def __init__(self,path_name):  
        self.train_images=None  
        self.train_labels=None  
         
        self.valid_images=None  
        self.valid_labels=None  
          
        self.test_images=None  
        self.test_labels=None  
         
        self.input_shape=(64,64,3)  
    
        self.path_name= path_name  
        
    def load(self,img_rows=IMAGE_SIZE,img_cols=IMAGE_SIZE,img_channels=3,num_classes=34):  
         images,labels=load_dataset(self.path_name)  
          
        train_images,valid_images,train_labels,valid_labels=train_test_split(images,labels,test_size=0.3,random_state=random.randint(0,100))  
        _,test_images,_,test_labels=train_test_split(images,labels,test_size=0.5,random_state=random.randint(0,100))  
          
        #根据keras库运行的后端要求修改图片通道顺序重组数据集  
        train_images=train_images.reshape(train_images.shape[0],img_rows,img_cols,img_channels)  
        valid_images=valid_images.reshape(valid_images.shape[0],img_rows,img_cols,img_channels)  
        test_images=test_images.reshape(test_images.shape[0],img_rows,img_cols,img_channels)  
          
        #输出训练集、验证集、测试集的数量  
        print(train_images.shape[0],'train samples')  
        print(valid_images.shape[0],'valid samples')  
        print(test_images.shape[0],'test samples')  
          
          
        #根据类别数量num_classes将类别标签进行one-hot编码使其向量化，在这里我们的类别只有34种，经过转化后标签数据变为多维  
        train_labels=np_utils.to_categorical(train_labels-1,num_classes)  
        valid_labels=np_utils.to_categorical(valid_labels-1,num_classes)  
        test_labels=np_utils.to_categorical(test_labels-1,num_classes)  
          
          
        #像素数据浮点化以便归一化  
        train_images=train_images.astype('float32')  
        valid_images=valid_images.astype('float32')  
        test_images=test_images.astype('float32')  
          
        #归一化  
        train_images/=255  
        valid_images/=255  
        test_images/=255  
          
        self.train_images=train_images  
        self.valid_images=valid_images  
        self.test_images=test_images  
        self.train_labels=train_labels  
        self.valid_labels=valid_labels  
        self.test_labels=test_labels  

class Model:  
    def __init__(self):  
        self.model=None  
        
    def build_model(self,dataset,num_classes=34):  
        self.model=Sequential()  
      
        self.model.add(Convolution2D(64,3,3,border_mode='same',input_shape = dataset.input_shape))  
        self.model.add(Activation('relu'))  
      
        self.model.add(Convolution2D(32,3,3))  
        self.model.add(Activation('relu'))  
      
        self.model.add(MaxPooling2D(pool_size=(2,2)))  
        self.model.add(Dropout(0.25))  
      
        self.model.add(Convolution2D(64,3,3,border_mode='same'))  
        self.model.add(Activation('relu'))  
          
        self.model.add(Convolution2D(64,3,3))  
        self.model.add(Activation('relu'))  
          
        self.model.add(MaxPooling2D(pool_size=(2,2)))  
        self.model.add(Dropout(0.25))  
          
        self.model.add(Flatten())#将数据押平一维化，从卷积层到全连接层的过渡  
        self.model.add(Dense(512))  
        self.model.add(Activation('relu'))  
        self.model.add(Dropout(0.5))  
        self.model.add(Dense(num_classes))  
        self.model.add(Activation('softmax'))#通过一个softmax映射为类别概率。我们这里是num_classes分类，因此最后的Dense层神经元数是num_classes  
          
        self.model.summary()  
    
    def train(self,dataset,batch_size=20,nb_epoch=40,data_augmentation=True):  
        #优化器,采用随机梯度下降法  
        sgd=SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)  
        self.model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])  
        if not data_augmentation:#不使用数据提升  
            self.model.fit(dataset.train_images,dataset.train_labels,batch_size=batch_size,nb_epoch=nb_epoch,validation_data=(dataset.valid_images,dataset.valid_labels),shuffle=True)#该函数shuffle参数用于指定是否随机打乱数据集  
        else:#使用实时数据提升    
            datagen=ImageDataGenerator(featurewise_center=False,samplewise_center=False,featurewise_std_normalization=False,samplewise_std_normalization=False,zca_whitening=False,rotation_range=20,width_shift_range=0.2,height_shift_range=0.2,horizontal_flip=True,vertical_flip=False)  
            datagen.fit(dataset.train_images)  
            self.model.fit_generator(datagen.flow(dataset.train_images,dataset.train_labels,batch_size=batch_size),samples_per_epoch=dataset.train_images.shape[0],nb_epoch=nb_epoch,validation_data=(dataset.valid_images,dataset.valid_labels))  
            
    MODEL_PATH='D:\\face detection\\data\\me.face.model.h5'   
  
    def save_model(self,file_path=MODEL_PATH):  
        self.model.save(file_path)  
      
    def load_model(self,file_path=MODEL_PATH):  
        self.model=load_model(file_path)  
        
    def evaluate(self,dataset):  
        score=self.model.evaluate(dataset.test_images,dataset.test_labels,verbose=1)  
        print("%s: %.2f%%" % (self.model.metrics_names[1],score[1]*100))  
        
    def face_predict(self,image):  
        if K.image_dim_ordering()=='th' and image.shape!=(1,3,IMAGE_SIZE,IMAGE_SIZE):  
            image=resize_image(image)  
            image=image.reshape((1,3,IMAGE_SIZE,IMAGE_SIZE))  
        elif K.image_dim_ordering()=='tf' and image.shape!=(1,IMAGE_SIZE,IMAGE_SIZE,3):  
            image=resize_image(image)  
            image=image.reshape((1,IMAGE_SIZE,IMAGE_SIZE,3))  
          
        image=image.astype('float32')  
        image/=255  
          
        result=self.model.predict_proba(image)  
        print('result:',result)  
          
        result=self.model.predict_classes(image)  
      
        return result[0]  
     
if __name__=='__main__':  
    dataset=Dataset('D:\\face detection\\data')  
    dataset.load()  
      
    model=Model()  
    model.load_model(file_path='D:\\face detection\\data\\me.face.model.h5')  
    model.evaluate(dataset)     
