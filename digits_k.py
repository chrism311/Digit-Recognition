import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
import cv2
from glob import glob

mnist = fetch_mldata('MNIST original')

X = np.array(mnist.data, 'int16')
y = np.array(mnist.target, 'int')

y = to_categorical(y)

#X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.1)

#Using own handwritten data
images, labels = [], []

for i in glob("/new_digits/*.png"):
	image = cv2.imread(i, 0)
	image = cv2.bitwise_not(image)
	images.append(image)

images = np.array(images).reshape([100, 784])/255.

digit = 0
while digit <= 9:
	for i in range(10):
		labels.append(digit)
	digit += 1
labels = to_categorical(labels)

model = Sequential()
model.add(Dense(50, input_dim=784, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, batch_size=128, epochs=20, verbose=1, validation_data= (images, labels))
score = model.evaluate(images, labels)
print 'Loss: %f' %score[0]
print 'Accuracy: %f' %score[1]
