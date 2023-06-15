#!/usr/bin/env python3
# import packages
import os
import myConfig as config
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Conv2D,Activation,Input,Add,Subtract,Concatenate,Conv2DTranspose,AveragePooling2D,Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LearningRateScheduler
import tensorflow.keras.backend as K
import numpy as np
import random
import tensorflow as tf
from utils import *
os.environ["CUDA_VISIBLE_DEVICES"]="7"
tf_device='/gpu:7'

# create CNN model
input_img=Input(shape=(None,None,3))

xn1=1-input_img

xn=Activation('relu')(xn1)
xn2=Conv2D(32,(3,3),dilation_rate=1,padding="same")(xn)

xn=Activation('relu')(xn2)
xn3=Conv2D(32,(3,3),dilation_rate=3,padding="same")(xn)

# PAB
x = Conv2D(32,(1,1),padding="same")(xn3)
x = Activation('relu')(x)
x12 = Conv2D(32,(1,1),padding="same")(x)

x = Conv2D(32,(2,2),padding="same")(xn3)
x = Activation('relu')(x)
x13 = Conv2D(32,(2,2),padding="same")(x)

x = Conv2D(32,(3,3),padding="same")(xn3)
x = Activation('relu')(x)
x14 = Conv2D(32,(3,3),padding="same")(x)

x = Conv2D(32,(4,4),padding="same")(xn3)
x = Activation('relu')(x)
x15 = Conv2D(32,(4,4),padding="same")(x)

x = Conv2D(32,(5,5),padding="same")(xn3)
x = Activation('relu')(x)
x16 = Conv2D(32,(5,5),padding="same")(x)

x_com1 = Add()([x12, x13])
x_com2 = Add()([x_com1, x14])
x_com3 = Add()([x_com2, x15])
x_com4 = Add()([x_com3, x16])
x=Activation('sigmoid')(x_com4)
xn4 = Multiply()([x,xn3])

xn=Activation('relu')(xn4)
xn=Conv2D(32,(3,3),dilation_rate=3,padding="same")(xn)

xn5 = Subtract()([xn2, xn])

xn=Activation('relu')(xn5)
xn6=Conv2D(32,(3,3),dilation_rate=1,padding="same")(xn)

x=Activation('relu')(input_img)
x10=Conv2D(32,(3,3),dilation_rate=1,padding="same")(x)

x=Activation('relu')(x10)
x11=Conv2D(32,(3,3),dilation_rate=3,padding="same")(x)

x = Conv2D(32,(1,1),padding="same")(x11)
x12 = Activation('relu')(x)

x = Conv2D(32,(2,2),padding="same")(x11)
x13 = Activation('relu')(x)

x = Conv2D(32,(3,3),padding="same")(x11)
x14 = Activation('relu')(x)

x = Conv2D(32,(4,4),padding="same")(x11)
x15 = Activation('relu')(x)

x = Conv2D(32,(5,5),padding="same")(x11)
x16 = Activation('relu')(x)

x_com11 = Add()([x12, x13])
x_com1 = Conv2D(32,(2,2),padding="same")(x_com11)
x_com21 = Add()([x12, x14])
x_com2 = Conv2D(32,(3,3),padding="same")(x_com21)
x_com31 = Add()([x12, x15])
x_com3 = Conv2D(32,(4,4),padding="same")(x_com31)
x_com41 = Add()([x12, x16])
x_com4 = Conv2D(32,(5,5),padding="same")(x_com41)
x_com5 = Concatenate()([x_com1, x_com2])
x_com6 = Concatenate()([x_com3, x_com5])
x_com7 = Concatenate()([x_com4, x_com6])
x_com8 = Concatenate()([x_com7, x11])

x=Activation('relu')(x_com8)
x2=Conv2D(32,(3,3),dilation_rate=3,padding="same")(x)

x3 = Subtract()([x10, x2])

x=Activation('relu')(x3)
x4=Conv2D(32,(3,3),dilation_rate=1,padding="same")(x)

x = Add()([x4, xn6])

x=Conv2D(32,(3,3),padding="same")(x)
x=Activation('relu')(x)

x5=Conv2D(3,(3,3),padding="same")(x)

model = Model(inputs=input_img, outputs=x5)

# load the data and normalize it
cleanImages=np.load(config.data)
print(cleanImages.dtype)
#cleanImages=cleanImages/255.0
cleanImages=cleanImages.astype('float32')

# define augmentor and create custom flow
aug = ImageDataGenerator(rotation_range=30, fill_mode="nearest")
def myFlow(generator,X):
    for batch in generator.flow(x=X,batch_size=config.batch_size,seed=0):
        noise=random.randint(0,55)
        trueNoiseBatch=np.random.normal(0,noise/255.0,batch.shape)
        noisyImagesBatch=batch+trueNoiseBatch
        yield (noisyImagesBatch,trueNoiseBatch)

# create custom learning rate scheduler
def lr_decay(epoch):
    initAlpha=0.001
    factor=0.5
    dropEvery=5
    alpha=initAlpha*(factor ** np.floor((1+epoch)/dropEvery))
    return float(alpha)
callbacks=[LearningRateScheduler(lr_decay)]

# create custom loss, compile the model
print("[INFO] compilingTheModel")
opt=optimizers.Adam(lr=0.001)
def custom_loss(y_true,y_pred):
    diff=y_true-y_pred
    res=K.sum(diff*diff)/(2*config.batch_size)
    return res
model.compile(loss=custom_loss,optimizer=opt)

# train
model.fit_generator(myFlow(aug,cleanImages),
epochs=config.epochs,steps_per_epoch=len(cleanImages)//config.batch_size,callbacks=callbacks,verbose=1)

# save the model
model.save('NIFBGDNet_Color.h5')
