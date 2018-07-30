import numpy as np
from Layer import Layer
from NeuralNetwork import NeuralNetwork
import matplotlib.pyplot as plt

#np.random.seed(1)
from sklearn import datasets, cross_validation, metrics # data and evaluation utils
from sklearn.model_selection import train_test_split
from matplotlib.colors import colorConverter, ListedColormap # some plotting functions
import itertools
import collections


def main():

    # Import the digits dataset
    digits = datasets.load_digits()
    y = np.zeros((digits.target.shape[0],10))
    y[np.arange(len(y)), digits.target] += 1

    # Divide the data into a train and test set.
    X_train, X_test, y_train, y_test = train_test_split(digits.data, y, test_size=0.4)

    # Normalizes data using one-hot encoding
    max = np.amax(X_train)
    X_train /= max
    X_test /= max

    # Creates neural network
    nn = NeuralNetwork()

    #TODO: Fix gradient descent when total_layers > 3

    # Adds the layers to the neural network
    nn.add_layer(64, bias=True)
    nn.add_layer(50, bias=True)
    # nn.add_layer(10, bias=True)
    nn.add_layer(10, bias=False)


    # Trains the model
    nn.train(X_train, y_train, iterations=300, learning_rate=0.5)

    # Predicts from the test set and calculates the accuracy
    predictions = nn.predict(X_test)
    print(predictions.shape)
    print(y_test.shape)
    accuracy = nn.calc_accuracy(y_test)
    nn.save("l1", min_acc=0.95)

    # Loads a pre-trained model and uses it to predict
    # nn.load("l1")
    # predictions = nn.predict(X_test)
    # accuracy = nn.calc_accuracy(y_test)




main()
