#Hand gesture recognition - deep learning project


#Importing libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

os.chdir("D:/_IRRI-SOUTH ASIA/personal projects/hand_gesture_recognition_project/dataset")

#Data Preprocessing
train = pd.read_csv("sign_mnist_train.csv")
test = pd.read_csv("sign_mnist_test.csv")

train_label=train["label"]
test_label = test["label"]

del train["label"]
del test["label"]

x = train.values/255 # normalized training set
y = test.values/255 # normalized testing set

x = x.reshape(-1,28,28,1)
y = y.reshape(-1,28,28,1)


#label binarizer - encoding labels into categories
label_binarizer = preprocessing.LabelBinarizer()
train_label = label_binarizer.fit_transform(train_label)
test_label = label_binarizer.fit_transform(test_label)


##Function to see the dataset
def show_image(num):
    num = num
    a = train.iloc[num,1:]
    b = np.array(a)
    c = b.reshape((28,28))
    l = train.iloc[num,0]
    plt.imshow(c,cmap='gray',label=l)
    return l

show_image(155)



#Builing CNN - deep learning model
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPool2D, Dropout,BatchNormalization, Flatten
from keras.callbacks import ReduceLROnPlateau

model = Sequential()
model.add(Conv2D(75,(3,3),strides=1,padding='same',activation='relu',input_shape=(128,128,1)))
model.add(BatchNormalization())
model.add(MaxPool2D(2,2))
model.add(Conv2D(75,(3,3),strides=1,padding='same',activation='relu'))
model.add(MaxPool2D(2,2))
model.add(Conv2D(75,(3,3),strides=1,padding='same',activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D(2,2))
model.add(Conv2D(75,(3,3),strides=1,padding='same',activation='relu'))
model.add(MaxPool2D(2,2))
model.add(Flatten())
model.add(Dense(units=512,activation='relu'))
model.add(Dense(units=1,activation='relu'))

model.compile(optimizer='adam', loss='mse',metrics=['accuracy'])

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',patience=2,verbose=1,factor=0.5,min_lr=0.00001) #From keras.callback import ReduceLROnPlateau #setting learning rate

#Training model
model.fit(x,trainy,batch_size=128,epochs=10,validation_data=(y,testy),callbacks=[learning_rate_reduction])


#Evaluating model - checking final accuracy
model.evaluate(y,test_label)

#Predicting
prediction = model.predict_classes(x)

for i in range(len(prediction)):
    if (prediction[i] >= 9 or prediction[i] >=25):
        prediction[i]+=1
        
prediction[:10]


#Saving model
model.save('hand_gesture.h5')


#Implementing model on web
import streamlit as st
st.write('''M''')



