# Strategical-Classifiers
Identification of Hand Written Digits
Dataset:
We work on MNIST dataset to work with KNN and Perceptron. The MNIST dataset is a database of handwritten digits, which has a training set of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image.
Methods used in Experiment:
KNN:
K-nearest neighbor classification algorithm is use to classify given images of handwritten number where we decide and classify an image based on Euclidean distance and voting method. KNN has used in statistical estimation and pattern recognition as a non-parametric technique.
Working with KNN we work with Euclidean distance and recognize the image with the help
 Of distance between two images. In addition to it we use cosine similarity function in which
 we recognize the digit by analyzing the angle computed by applying cosine function on 
Images.

Perceptron : 
Perceptron is a computer model or computerized machine devised to represent or simulate 
The ability of the brain to recognize and discriminate. With the help of this we recognize 
numbers given as a dataset

 
 
A perceptron image in shown above.
A perceptron use an activation function for classification. In this experiment we work and 
analyze the behavior of three types of activation function named as:
•	Linear Classifier
•	Sigmoid Activation function using gradient descent 
•	Tan Activation function using gradient descent 
Gradient descent is actually use to minimize the error while computing weights. 
Gradient descent is an optimization algorithm used to minimize some function by 
iteratively moving in the direction of Steepest descent as defined by the negative 
Of the gradient. We use Gradient descent to update the parameters of our model.

Experiments:

For KNN:
 We set different values of k to predict the result. 
Got maximum accuracy for k = 30 and got 96% accuracy for the above dataset.
Code is provided with this document.
Run code for ten times to get result.

For Perceptron:

60,000 images were used to train the data for each number and then tested on ten thousand
Images where epoch was set 20.
Sigmoid:

Gradient to minimize the  error in computed weights

temp =  image * (LEARNING_RATE * (label - classify))
    weighted_arr = weighted_arr + temp

    for j in range(785):
        gradient = LEARNING_RATE * (label - classify) * (1-classify) * classify * weighted_arr[j]
        weighted_arr[j] = weighted_arr[j] + gradient

    return weighted_arr

calculating dot product and using sigmoid.
x = np.dot(weightage , image)
return 1 / (1 + math.exp(-x))		
Same process is done with tan as given in the code files attached.

Linear Classifier :

    output = np.dot(weightage , image)
    if (output > 0):
        return 1
    else:
        return -1

A threshold mechanism as use in linear classification is used here to predict either the 
Given image is same for which we are prediction or not and then weights are learned.

def Train_Weightages(image , weighted_arr , label):


    classify = callabel(weighted_arr, image)

    temp = (LEARNING_RATE * (label - classify)) * image
    new_weight = weighted_arr + temp

    return new_weight

See more and full logic of code in linearClassification.py files attached with this report.

Conclusion
Working and experimenting with all these methods KNN took a  lot of time to learn
Where other methods took less time as compared and give better results.

Linear classifiers do not work on complex data sets and accuracy was only 20 percent 
Sigmoid gives 85 percent accuracy.
Tan also work well and gives 95 percent accuracy.
		
