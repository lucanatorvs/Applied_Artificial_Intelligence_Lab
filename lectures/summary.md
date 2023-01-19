# AI Things to remember:

**Batch, mini-batch and stochastic gradient descent** are  the method of updating the parameters by computing the gradient of the loss function with respect to the parameters:
 
- batch = using the entire training set.
- mini-batch = using a subset of the training set.
- stochastic  = using a single training example.

**Binary logistic regression** is supervised learning where the output is a binary value. The _sigmoid function_ is used to transform the output to a value between 0 and 1.

**Logistic regression for multiple classes** outputs a vector of probabilities. The *softmax function* is used to transform the output.

**Supervised learning** is when the dataset includes correct answers and the goal is to train a model to predict them.

**Unsupervised learning** is when the dataset does not include correct answers and the goal is to find patterns in the data.

**Linear regression** is when the goal is to find a line or plane that minimizes the loss function, and how it can be applied to single or multiple variables.

**Polynomial regression**, which is a type of linear regression that involves adding higher order polynomials to the features and training a linear regression model on the new features.

The **normal equation** is a formula that gives the exact solution to the linear regression problem.

**Feature scaling** involves transforming the features to have similar scales, in order to improve the performance of the model.

**Overfitting** occurs when the model is too complex and does not generalize well to new data. It can be coused by to high a degree of polynomial. The model fits very well to the training data, but does not generalize well to new data.

**Binary logistic regression** is a type of supervised learning for binary outputs, it uses the cross entropy loss function and the sigmoid function to transform the output. It is used for classification problems.

**Classification** is a type of supervised learning where the output is a discrete value. The output is a probability distribution over the possible classes.

**Logistic regression** is a type of classification algorithm that transforms the output to a probability distribution over the possible classes.

**Regression** is used for continuous outputs, while **classification** is used for discrete outputs.

diferent output functions for different types of problems:

- **Sigmoid function** for binary classification.
- **Softmax function** for multi-class classification.
- **Linear function** for regression.

These are used to transform the output of a model into a range of values or probabilities.

**Cross entropy** is a loss function used for classification problems. It is used to measure the difference between the predicted probability distribution and the true probability distribution.

**regularization** is a technique used to reduce overfitting. It involves adding a penalty term to the loss function.

**L1 and L2 regularization** adds a penalty based on the absolute or squared values of the model parameters, respectively.

**K-nearest neighbor** is a type of supervised learning algorithm for classification and regression. It is used to predict the class of a new data point by looking at the classes of the k nearest data points.

- Look at the k number of nearest neighbors and classify the new point based on the majority of the neighbors.

Minkowski distance is a generalization of the Euclidean distance and the Manhattan distance.

- **Euclidean distance** is the most common distance metric. It is the straight line distance between two points.
- **manhattan distance** is the distance between two points measured along axes at right angles. (city block distance)

chooseing the right value for k is important, if k is too small, the model will be too sensitive to noise. If k is too large, the model will be too simple and will not capture the local structure of the data.

regression is done by taking the average or mean of the k nearest neighbors.

_advantages_: simple, easy to understand, easy to implement, versatile, no assumptions about data, fast training.
_disadvantages_: high memory usage, slow prediction, sensitive to irrelevant features, sensitive to the scale of the data.

a **support vector machine** is a supervised learning algorithm that can be used for classification or regression. It is used to find a hyperplane that best separates the data. The hyperplane is the decision boundary that separates the classes.

_Support vectors_ are the data points that are closest to the hyperplane. They are the points that determine the position and orientation of the hyperplane.

_Margin_ is the distance between the hyperplane and the support vectors. The goal is to maximize the margin.

add extra demantions to the data to make it linearly separable.

**decision tree** is a supervised learning algorithm that can be used for classification. It is used to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features.

_the root node_ is the top node in the tree. It is the first node that is used to make a decision. split the data based on the feature that gives the largest information gain.

Roatating the data, _feature engineering_ can be used to make the data linearly separable.

you can prevent _overfitting_ by limiting the depth of the tree, or by setting a minimum number of samples required at a leaf node.

_purity_ and _entropy_ in relation to a decision tree is a measure of how well the data is split.

_impurity_ is the probability that a randomly chosen element from the set will be incorrectly labeled if it is randomly labeled according to the distribution of labels in the subset.

_entropy_ is a measure of impurity. It is the average information content of the labels in a data set.

**regression tree** is a supervised learning algorithm that can be used for regression. It is used to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features.

- take the mean of the target values in the leaf node.

The **random forest** algorithm takes many decision trees and combines them to make a more accurate and stable prediction. Every decision tree in the random forest is trained on a different subset of the data.

_bagging_, and _out of bag estimators_ are used to get a more accurate prediction, if you do it with a(manny) disicion tree(s), you get a random forest.

use the ensemble module in sklearn

**voting classifiers** are a way of combining the predictions of several models to get a better prediction. It is used to combine the predictions of different models. try different models and take the one that performs the best.

**loss function** is a function that is used to measure how well a model performs on a given dataset. It is used to optimize the model parameters.

==============================

**challenges in machine learning**:

- not enough data
- data innconsistencies/poor quality
- nonrepresentative data
- irrelevant features

the training data should be representative of data u use in production.

## Why do we need to split our data into different sets?

all data splits into training, validation and test data: 70/30 or 80/20 train/test split. than the training data is split into training and validation data.

the difference between a **validation set** and a **test set** is that the validation set is used to tune the hyperparameters of the model, while the test set is used to evaluate the final model.

the test set is used to test the model and is never "seen" by the model.

**confusion matrix** for a binary classifier is a table that is used to describe the performance of a classification model on a set of test data for which the true values are known.

it shows the true positives, true negatives, false positives and false negatives:

- _true positive_: the model predicted a positive and it was positive **(TP)**
- _true negative_: the model predicted a negative and it was negative **(TN)**
- _false positive_: the model predicted a positive and it was negative **(FP)**
- _false negative_: the model predicted a negative and it was positive **(FN)**

### accuracy:

the number of correct predictions divided by the total number of predictions

$$\frac{TP + TN}{TP + TN + FP + FN}$$

### precision:

the number of correct positive predictions divided by the total number of positive predictions

$$\frac{TP}{TP + FP}$$

### recall:

the number of correct positive predictions divided by the total number of positive values in the test data

$$\frac{TP}{TP + FN}$$

### f1 score:

the harmonic mean of precision and recall.

$$\frac{TP}{TP + \frac{1}{2}(FP + FN)}$$

### selectivety/specificity:

the number of correct negative predictions divided by the total number of negative values in the test data, it gives similar information to recall.

$$\frac{TN}{TN + FP}$$

**ROC-AUC** is a performance metric for binary classification problems. It is a probability curve and AUC represents the degree or measure of separability. It tells how much model is capable of distinguishing between classes.

we plot the true positive rate (recall) against the false positive rate (1 - specificity). we want the area under the curve to be as big as possible.

## What does a learning curve look like?

**learning curve** is a plot of the model's performance on the training set and the validation set as a function of progesion through the training process (epoch).

epoch is the number of iterations

we plot the loss and error on the training set and the validation set.

**error** is the number of misclassifications

**loss** is the output of the cost function

we find out if we are under or overfitting by looking at the learning curve. if the training error is high and the validation error is low, we are underfitting. if the training error is low and the validation error is high, we are overfitting.

**parameters** are the model's internal parameters that are learned during training. they are the weights and biases.

**hyper-parameters** are the parameters that are not learned during training. they are the number of layers, the number of neurons in each layer, the learning rate, the activation function, the optimizer, the batch size, the number of epochs, etc.

**K-fold cross validation** is a technique for evaluating machine learning models. it is used to split the data into k subsets, and then train and evaluate the model k times, each time using a different subset for evaluation and the remaining k-1 subsets for training.

**regularization** is a technique used to prevent overfitting. it is used to penalize the model for having large weights. it is used to reduce the variance of the model. and it keeps the slope of the loss function small

a **batch** is a subset of the training data. the batch size is the number of samples in a batch.

**epoch** is the number of iterations. one epoch is when an entire dataset is passed forward and backward through the neural network once.

**early stopping** is a regularization technique that is used to prevent overfitting. it is used to stop the training process when the validation error reaches a local minimum. but it may not be the global minimum.

==============================

## nural networks (MLP)

A neural network is a mathematical model of a biological neural network. It is a network of artificial neurons.

A nuron is a mathematical function that takes a set of inputs, performs some computation on them, and produces an output.

a layer is a set of neurons. a neural network is a set of layers.

a perceptron is a single layer neural network.

waights and biases are the parameters of a neural network.

### activation functions

a activation function allows us to model non-linear relationships between the inputs and the outputs.

- _step function_: it is a function that returns 0 or 1 depending on the input. it is not differentiable and it is not used in neural networks.
- _sigmoid function_: it is a function that returns a value between 0 and 1. it is used in the output layer of a binary classification problem. it is not used in hidden layers.
- _tanh function_: it is a function that returns a value between -1 and 1. it is used in hidden layers.
- _rectified linear unit (ReLU) function_: it is a function that returns 0 if the input is negative and the input if the input is positive. it is used in hidden layers.
- _leaky ReLU function_: it is a function that returns 0.01x if the input is negative and the input if the input is positive. it is used in hidden layers. it is used to solve the dying ReLU problem.
- _softmax function_: it is a function that returns a probability distribution. it is used in the output layer of a multi-class classification problem. it is not used in hidden layers.

### loss functions

the los function measures how well the model performs on a given dataset. it is used to optimize the model parameters.

- _mean squared error (MSE) loss function_: the result of this function is the average of the squared differences between the predicted values and the target values. it is used in regression problems.
- _cross-entropy loss function_: the result of this function is the average of the negative log of the predicted values. it is used in classification problems.

softmax, it is used for the output layer.

cross entropy, it is used for multi-class classification.

### cost functions

the cost function is the loss function plus the regularization term. it is used to optimize the model parameters.

### optimization algorithms

- _gradient descent_: it is an optimization algorithm that is used to find the minimum of a function. it is used to optimize the model parameters.

optput computation is the output of a nuron. it is the weighted sum of the inputs plus the bias.

activation computation is the output of an activation function. it is the output of a nuron passed through an activation function.

fully connected / dense layer is a layer where each neuron is connected to every neuron in the previous layer.

the goal of training is to minimize the cost function. we do this by updating the weights and biases.

backpropagation is the process of updating the weights and biases, we take the dirivative of the weights of the nurons in the output layer, and then we take the derivative of the weights of the nurons in the hidden layer, and so on. we check how small chainges in the waights and biases affect the cost function. we then update the weights and biases in the direction that reduces the cost function. for this we need a activation function that is differentiable, so we can not use the step function.

ReLU is a activation function that is not differentiable at 0, but we can still use it and is is computationally efficient.

**learning rate** is a hyper-parameter that is used to control how much we update the weights and biases. it is used to prevent overshooting the minimum. but if it is too small, it will take a long time to converge.

**dropout** is a regularization technique that is used to prevent overfitting. it is used to randomly drop out some neurons during training. it is used to reduce the variance of the model. and it keeps the slope of the loss function small. basicly it makes sure the model does not rely on any one neuron.

### vanishing gradient problem

when you have a deep neural network, the gradient can get very small. this makes it hard for the network to learn. there are a few ways to fix this:

- use a different activation function (tanh, relu, leaky relu), but the mpre complicated ones are more computationally expensive
- use batch normalization
- use residual networks ??
- improve weight initialization

### batch normalization

normalize the inputs to each layer. this makes the network more robust to different initializations. it also makes it easier to use a higher learning rate.

`keras.layers.BatchNormalization()` is a layer that normalizes the inputs to each layer.

### optimizers

#### momentum optimization

- keeps track of the direction of the previous gradients
- adds a fraction of the previous gradient to the current gradient
- this helps the algorithm to jump out of local minima

#### stochastic gradient descent (SGD)

- instead of computing the gradients based on the full training set at every step, it computes the gradients based on a random subset of the training set
- this makes the algorithm much faster, but it also makes it a lot less regular
- this can help the algorithm jump out of local minima, but it can also prevent it from converging to the global minimum

#### adaptive gradient optimization (AdaGrad)

- divides the learning rate by the square root of the sum of the squares of the gradients up to that point
- this makes the algorithm converge faster

#### adam optimization

- combines momentum optimization and adaptive gradient optimization
- it also keeps track of an exponential decaying average of past gradients

## convolutional neural networks (CNN)

a convolutional neural network is a neural network that is used for image classification.

### convolutional layer

a convolutional layer is a layer that is used to extract features from the input image.

- _stride_: the number of pixels to shift the filter at each step
- _depth_: the number of filters
- _padding_: the number of pixels to add to the input image
  - _valid padding_: no padding
  - _same padding_: the output image has the same size as the input image
- _filter size_: the size of the filter/kernel
- _filter depth_: the number of channels in the input image
- _pooling layer_: a layer that is used to reduce the size of the input image
  - _max pooling layer_: a layer that is used to reduce the size of the input image by taking the maximum value of each patch
  - _average pooling layer_: a layer that is used to reduce the size of the input image by taking the average value of each patch


### data augmentation

- flip the image horizontally
- shift the image up, down, left, or right
- rotate the image
- zoom in on the image
- shear the image

this is used to prevent overfitting, by manufaturing more data.

### transfer learning (learning from pretrained models)

- use a pretrained model as a starting point
- you can use the pretrained model as a fixed feature extractor

## principal component analysis (PCA)

There are several applications of principal component analysis (PCA) in machine learning. Some of the main applications are:

- Data visualization: PCA can be used to reduce the dimensionality of high-dimensional data and visualize it in a lower-dimensional space. This can be useful for understanding patterns and relationships in the data.
- Noise reduction: PCA can be used to remove noise from a dataset by projecting the data onto a lower-dimensional subspace that preserves the most variance.
- Data compression: PCA can be used to compress a dataset by projecting it onto a lower-dimensional subspace and storing only the principal components. This can be useful for reducing storage and computational requirements.
- Anomaly detection: PCA can be used to identify anomalies or outliers in a dataset by analyzing the reconstruction error of the data after projecting it onto a lower-dimensional subspace.
- Feature selection: PCA can be used to select the most important features in a dataset by selecting the principal components that preserve the most variance. This can be useful for improving the performance of machine learning algorithms.

### k-means

randomly initialize k centroids. assign each data point to the nearest centroid. recompute the centroids. repeat until the centroids don't change.
