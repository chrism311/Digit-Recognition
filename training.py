from struct import unpack
from numpy import zeros, uint8, float32, vstack, array, append, where
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import cv2

#Code found online to unpack/read raw data files
def unpack_data(imagefile, labelfile):
	"""Read input-vector (image) and target class (label, 0-9) and return
	it as list of tuples.
	"""
	# Open the images with gzip in read binary mode
	images = open(imagefile, 'rb')
	labels = open(labelfile, 'rb')
	# Read the binary data
	# We have to get big endian unsigned int. So we need '>I'
	# Get metadata for images
	images.read(4)  # skip the magic_number
	number_of_images = images.read(4)
	number_of_images = unpack('>I', number_of_images)[0]
	rows = images.read(4)
	rows = unpack('>I', rows)[0]
	cols = images.read(4)
	cols = unpack('>I', cols)[0]

	# Get metadata for labels
	labels.read(4)  # skip the magic_number
	N = labels.read(4)
	N = unpack('>I', N)[0]

	if number_of_images != N:
		raise Exception('number of labels did not match the number of images')
	# Get the data
	x = zeros((N, rows, cols), dtype=float32)  # Initialize numpy array
	y = zeros((N, 1), dtype=uint8)  # Initialize numpy array
	for i in range(N):
		if i % 1000 == 0:
			print("i: %i" % i)
		for row in range(rows):
			for col in range(cols):
				tmp_pixel = images.read(1)  # Just a single byte
				tmp_pixel = unpack('>B', tmp_pixel)[0]
				x[i][row][col] = tmp_pixel
		tmp_label = labels.read(1)
		y[i] = unpack('>B', tmp_label)[0]
	return x, y

#Reads and stores data from binary files in variables at a specified percentage split
def data(p):
	images60k, labels60k = unpack_data('train-images-idx3-ubyte', 'train-labels-idx1-ubyte')
	images10k, labels10k = unpack_data('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte')
	X = vstack((images60k, images10k))
	y = vstack((labels60k, labels10k))
	X = X.reshape([70000, 784])
	X = X / 255.
	y = y.reshape([70000,])
	X = X[:int(70000*p)]
	y = y[:int(70000*p)]
	return X, y

#Outputs # of misclassified and confusion matrix
def conf_matrix(true, prediction):
	k = true == prediction
	index = where(k == False)
	misclass = len(index[0])
	print "Correct: %d, Misclassified: %d" % ((len(true)-misclass), misclass)
	print confusion_matrix(true, prediction, labels=[0,1,2,3,4,5,6,7,8,9])

#Trains classifiers
def train(clf_1, clf_2, clf_3, i):
	p= i 
	print 'Training...'
	while p < 1.0:
		x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=1-p)
		for clf in (clf_1, clf_2, clf_3):
			clf.fit(x_train, y_train)
			train_pre = clf.predict(x_train)
			test_pre = clf.predict(x_test)
			train_score = 100*accuracy_score(y_train, train_pre)
			test_score = 100*accuracy_score(y_test, test_pre)
			print "***", clf.__class__.__name__, "Split: %f, Training Perc: %f, Test Perc: %f" % (p, train_score, test_score)
			print "Training CF:"
			conf_matrix(y_train, train_pre)
			print "Testing CF:"
			conf_matrix(y_test, test_pre)
		p += .05
		print "\n\n"

if __name__== '__main__':
	images, labels = data(1)
	clf_extra = ExtraTreesClassifier()
	clf_mlp = MLPClassifier()
	clf_knn = KNeighborsClassifier()
	train(clf_extra, clf_mlp, clf_knn, .3)


