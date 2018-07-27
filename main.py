import numpy as np
from Layer import Layer
from NeuralNetwork import NeuralNetwork
import matplotlib.pyplot as plt

np.random.seed(1)
from sklearn import datasets, cross_validation, metrics # data and evaluation utils
from matplotlib.colors import colorConverter, ListedColormap # some plotting functions
import itertools
import collections


def main():

    # Import the digits dataset
    digits = datasets.load_digits()
    T = np.zeros((digits.target.shape[0],10))
    T[np.arange(len(T)), digits.target] += 1

    # Divide the data into a train and test set.
    X_train, X_test, T_train, T_test = cross_validation.train_test_split(digits.data, T, test_size=0.4)
    # Divide the test set into a validation set and final test set.
    X_validation, X_test, T_validation, T_test = cross_validation.train_test_split(X_test, T_test, test_size=0.5)

    max = np.amax(X_train)

    #X_train /= max
    #print(X)

    nn = NeuralNetwork()

    nn.add_layer(64, bias=True)
    nn.add_layer(100, bias=True)
    nn.add_layer(10, bias=False)

    train = np.array(([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]]))
    train_y = np.array([[0,0],[1,1],[1,0],[0,1]])
    #nn.predict(test)
    test = np.array(([1,1,0]))

    #prediction = nn.predict(X_train)
    #nn.display()

    #nn.display()
    error = nn.train(X_train, T_train, iterations=20, learning_rate=0.1)
    prediction = nn.predict(X_train[55,:])
    nn.display()
    print(prediction)
    print(T_train[55,:])
    #print("Approx grads: " + str(nn.check_gradient(X_train, T_train)))

    #cost = nn.cost(test, test_y)
    #nn.display()

    #print(error)
    #print(cost)




main()
