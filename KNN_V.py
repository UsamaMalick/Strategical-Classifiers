from keras.datasets import mnist
from keras.layers.convolutional import Conv2D
from keras import backend as k
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
from collections import Counter
import time


def load_data():
	train_size = 60000
	test_size = 10000
	v_length = 784

	(X_train, y_train), (X_test, y_test) = mnist.load_data()

	format(X_train.shape)
	format(X_test.shape)
	format(X_train.shape[0])
	format(X_test.shape[0])

	# reshape the dataset
	X_train = X_train.reshape(train_size, v_length)
	X_test = X_test.reshape(test_size, v_length)
	X_train = X_train.astype("float32")
	X_test = X_test.astype("float32")
	X_train /= 255
	X_test /= 255

	return X_train , y_train , X_test , y_test



class simple_knn():
    #"a simple kNN with L2 distance"

    def __init__(self):
        pass

    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1):
        dists = self.compute_distances(X)
        # print("computed distances")

        num_test = dists.shape[0] # give rows of the dists.

        y_pred = np.zeros(num_test)

        for i in range(num_test):
            k_closest_y = []
            labels = self.y_train[np.argsort(dists[i,:])].flatten()
            # find k nearest lables
            k_closest_y = labels[:k]

            # out of these k nearest lables which one is most common
            # for 5NN [1, 1, 1, 2, 3] returns 1
            # break ties by selecting smaller label
            # for 5NN [1, 2, 1, 2, 3] return 1 even though 1 and 2 appeared twice.
            c = Counter(k_closest_y)
            y_pred[i] = c.most_common(1)[0][0]

        return(y_pred)


    def cosine(self , X , k=1):

        List = []
        k_closest_y = []

        for row in self.X_train:
            List.insert(dot(X , row)/(np.linalg.norm(X) * np.linalg.norm(row)))
        
        List.sort()
        k_closest_y = List[:k]

        occurence_count = Counter(k_closest_y) 
        prediction = occurence_count.most_common(1)[0][0] 

        return self.y_train[angels[prediction]]; 

    def compute_distances(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]

        dot_pro = np.dot(X, self.X_train.T)
        sum_square_test = np.square(X).sum(axis = 1)
        sum_square_train = np.square(self.X_train).sum(axis = 1)
        dists = np.sqrt(-2 * dot_pro + sum_square_train + np.matrix(sum_square_test).T)

        return(dists)  



(trainData, trainLabels, testData, testLabels) = load_data()


classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
num_classes = len(classes)
samples = 20

for y, cls in enumerate(classes):
    idxs = np.nonzero([i == y for i in trainLabels])
    idxs = np.random.choice(idxs[0], samples, replace=False)
    for i , idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples, num_classes, plt_idx)
        plt.imshow(trainData[idx].reshape((28, 28)))
        plt.axis("off")
        if i == 0:
            plt.title(cls)
        

plt.show()

plt.imshow(testData[2540].reshape((28, 28)))
plt.show()


k=30
classifier = simple_knn();
classifier.train(trainData , trainLabels)



predictions = []
cosine_prediction = []



print('From angle between two vectors.')
for row in X_test:
    print('Computing' + str(i+1) + '/' + str(len(X_test)) + '...')
    tic = time.time()
    predts = kNN.cosine(row, k)
    cosine_prediction.insert(predts)
    toc = time.time()
    print('Completed this batch in ' + str(toc-tic) + 'Secs.')

print('Now from eucladian(L2)')



for row in testData:
   	print('Computing batch' + str(i+1) + '/' + str(len(testData)) + '...')
   	tic = time.time()
   	predts = classifier.predict(row, k)
   	predts = kNN.cosine(X_test, k)
   	toc = time.time()
   	cosine_prediction.insert(predts)
   	predictions = predictions + list(predts)
   	print('Completed this batch in ' + str(toc-tic) + 'Secs.')

print('Completed predicting the test data.')

predics = int((len(predictions))/10)

out_file = open("predictions.csv", "w")
out_file.write("ImageId,Prediction,Label\n")

wrong = 0
for i in range(len(predictions)):
	predic_num = int(predictions[i])
	out_file.write(str(i+1) + ',' + str(predic_num)+ ',' + str(testLabels[i])  + '\n')
	if(predic_num != testLabels[i]):
		wrong = wrong + 1;


correct = (len(predictions)-wrong)
percent = (correct/len(predictions))*100

out_file.write("accuracy ,") 
out_file.write(str(percent) + '%')
out_file.close()





























