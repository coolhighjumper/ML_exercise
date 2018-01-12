import numpy as np
#for reproducibility
np.random.seed(123)
from keras.models import Sequential
from keras.layers import Dense,Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils

from keras.datasets import mnist
(X_train,y_train),(X_test,y_test)=mnist.load_data()
#(60000, 28, 28)

from matplotlib import pyplot as plt
plt.imshow(X_train[0])

X_train=X_train.reshape(X_train.shape[0],28,28,1)
X_test=X_test.reshape(X_test.shape[0],28,28,1)
#print(X_train.shape)


X_train=X_train.astype('float32')
X_test=X_test.astype('float32')
X_train/=255
X_test/=255

#flatten y_train and y_test into categorical class
Y_train=np_utils.to_categorical(y_train,10)
Y_test=np_utils.to_categorical(y_test,10)

#construct model
model=Sequential()
#first-layer -> convolution layer
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
#second-layer -> convolution layer
model.add(Conv2D(32,(3,3),activation='relu'))
#third-layer -> maxpooling layer
model.add(MaxPooling2D(pool_size=(2,2)))
#fourth-layer -> dropout layer
model.add(Dropout(0.25))
#flatten
model.add(Flatten())

#fully-connected neural network
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))

#define loss fct and optimizer
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#fitting
model.fit(X_train,Y_train,batch_size=32,epochs=10,verbose=1)

#using testing data
score=model.evaluate(X_test,Y_test,verbose=0)
predict=model.predict(X_test)

import pandas as pd
test=pd.read_csv('test.csv')
test=np.array(test)
test=test.reshape(test.shape[0],28,28,1)
predict=model.predict(test)
ans=[]
for i in range(test.shape[0]):
    ans.append(np.where(predict[i]==predict[i].max())[0][0])
np.savetxt('result.csv',ans,fmt='%i',delimiter=',')
