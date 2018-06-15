import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import numpy as np
from sklearn.datasets import fetch_mldata
import cv2
from glob import glob

mnist = fetch_mldata('MNIST original')

X = np.array(mnist.data, 'int16')/255.		#Normalizing the matrices
y = np.array(mnist.target, 'int')

y = to_categorical(y)				#Converting into binary matrix

#Using own handwritten data
images, labels = [], []

for i in glob("new_digits/*.png"):
	image = cv2.imread(i, 0)
	image = cv2.bitwise_not(image)
	images.append(image)

images = np.array(images).reshape([100, 784])/255.

digit = 0					#Creating the labels for my images
while digit <= 9:
	for i in range(10):
		labels.append(digit)
	digit += 1
labels = to_categorical(labels)

model = Sequential()										#Defining model
model.add(Dense(50, input_dim=784, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])		#Compiling model
model.fit(X, y, batch_size=512, epochs=35, verbose=1, validation_data= (images, labels))
score = model.evaluate(images, labels)								#Evaluating my images
print 'Loss: %f' %score[0]
print 'Accuracy: %f' %score[1]
