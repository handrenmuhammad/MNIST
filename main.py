from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical

from keras import models
from keras import layers

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# reshape the data into a 1D vector
x_train = x_train.reshape(60000, 28*28)
x_test = x_test.reshape(10000, 28*28)

# normalize the data to be between 0 and 1
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255


# one-hot encode the labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model=models.Sequential()
model.add(layers.Dense(512,activation='relu',input_shape=(28*28,)))
model.add(layers.Dense(10,activation='softmax'))

print(model.summary())

model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train,y_train,epochs=5,batch_size=128)

test_loss,test_acc=model.evaluate(x_test,y_test)
print("test_loss: ",test_loss,"\n","test_acc: ",test_acc)
