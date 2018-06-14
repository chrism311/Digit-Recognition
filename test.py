from training import data, conf_matrix
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, accuracy_score
import cv2
import matplotlib.pyplot as plt
from glob import glob
import numpy as np

X_train, y_train = data(1)
#X_train, y_train = shuffle(X_train, y_train)

images = []
labels = [] 

#Reads custom handwritten digits
for pic in glob('/new_digits/*.png'):
	image = cv2.imread(pic, 0)
	image = cv2.bitwise_not(image)
	images.append(image)

#Reshaping images and normalizing
images = np.array(images).reshape([100, 784])/ 255.
plt.imshow(X_train[0].reshape(28,28),cmap='gray')
plt.show()
plt.imshow(images[50].reshape(28,28), cmap='gray')
plt.show()
#Makes labels for custom handwritten digits
digit = 0
while digit <= 9:	
	for x in range(10):
		labels.append(digit)
	digit +=1

clf_extra = ExtraTreesClassifier(n_estimators=20)
clf_mlp = MLPClassifier(hidden_layer_sizes=(150,))
clf_knn = KNeighborsClassifier(n_neighbors=15)

for clf in (clf_extra, clf_mlp, clf_knn):
	clf.fit(X_train, y_train)
	prediction = clf.predict(images)
	acc = accuracy_score(labels, prediction)
	print '***', clf.__class__.__name__, "Accuracy for custom digits: %f" % (100*acc)
	conf_matrix(labels, prediction)
