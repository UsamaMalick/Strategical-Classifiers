import math
import random
import numpy as np
from keras.datasets import mnist
import time

No_Of_Image = 60000
No_Of_Column = 28
No_Of_Row = 28
test_size = 10000
IMAGE_SIZE = 785


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



(TrainData, Training_Label_Arrays, TestData, Test_Label_Arrays) = load_data()
Training_Image_Arrays = np.arange(47100000 ,dtype='f').reshape(No_Of_Image , 785)
Test_Image_Arrays = np.arange(7850000 , dtype='f').reshape(test_size , 785)


def Weighted_Arr():
    weighted_arr = np.arange(785 , dtype='f')
    for x in range(0, 785):
        randomNUM = float(random.uniform(0 , 0.5))
        weighted_arr[x] = randomNUM

    return weighted_arr

def PutingBiasTerm(bias = 1):
    for x in range(0 , No_Of_Image):
        j = 0
        Training_Image_Arrays[x][j] = bias
        if(x<test_size):
            Test_Image_Arrays[x][j] = bias
        for y in range(0 , 784):
            j = j+1
            Training_Image_Arrays[x][j] = TrainData[x][y]
            if(x<test_size):
                Test_Image_Arrays[x][j] = TestData[x][y]

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def callabel(weightage, image):

    output = np.dot(weightage , image)
    return sigmoid(output)


def Train_Weightages(image , weighted_arr , label):

    classify = callabel(weighted_arr, image)

    temp =  image * (LEARNING_RATE * (label - classify))
    weighted_arr = weighted_arr + temp

    for j in range(785):
        gradient = LEARNING_RATE * (label - classify) * (1-classify) * classify * weighted_arr[j]
        weighted_arr[j] = weighted_arr[j] + gradient

    return weighted_arr

new_labels = [0] * No_Of_Image

def call_perception(num , weighted_arr):

    for x in range(No_Of_Image):
        if(Training_Label_Arrays[x] == num):
            new_labels[x] = 1
        else:
            new_labels[x] = -1

    for k in range(1):
        print ( "Epoch : " ,  k)
        for x in range(60000):
            weighted_arr = Train_Weightages(Training_Image_Arrays[x] , weighted_arr , new_labels[x])

    wrong = correct = 0

    for x in range(10000):

        decision = (callabel(weighted_arr, Test_Image_Arrays[x]) )
        print("DECISION : " , decision)
        if(decision > 0.7 and  decision < 1.0):
            correct = correct + 1
        else:
            wrong = wrong + 1


    print("Wrong Predictions :" ,wrong)
    print("Correct Predictions :" , correct)
    total = wrong + correct;
    accuracy = (correct/total)*100
    print("Accuracy is " , accuracy , "%")
    print("Weights for : " , num)
    print(weighted_arr)


BIAS = 1
LEARNING_RATE = 0.00001
PutingBiasTerm(BIAS)


print("Learning")
tic = time.time()
for train_number in range(10):
    print ("FOR " , train_number)
    weighted_arr = Weighted_Arr()
    call_perception(train_number , weighted_arr)

toc = time.time()
print('Completed this batch in ' + str(toc-tic) + ' Secs.')

