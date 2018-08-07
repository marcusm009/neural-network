import numpy as np
from Layer import Layer
from NeuralNetwork import NeuralNetwork
import matplotlib.pyplot as plt

#np.random.seed(1)
from sklearn import datasets, metrics # data and evaluation utils
from sklearn.model_selection import train_test_split
from matplotlib.colors import colorConverter, ListedColormap # some plotting functions
import itertools
import collections

#np.set_printoptions(precision=3)

#TODO: Make an interface to see predictions
#TODO: Make an interface to allow drawing
#TODO: Fix gradient checking
#TODO: Test on more training sets
#TODO: Refactor and explain code better
#TODO: Add a webscraping and cleaning algorithm

def main():

    # Import the digits dataset
    digits = datasets.load_digits()
    y = np.zeros((digits.target.shape[0],10))
    y[np.arange(len(y)), digits.target] += 1

    # Divide the data into a train and test set.
    X_train, X_test, y_train, y_test = train_test_split(digits.data, y, test_size=0.4)

    # Shows the first image of the training set
    # plt.gray()
    # plt.matshow(digits.images[0])
    # plt.show()

    # Normalizes data using one-hot encoding
    max = np.amax(X_train)
    X_train /= max
    X_test /= max

    # Creates neural network
    nn = NeuralNetwork(initial_weight=0.075)

    # Initializes layer scheme
    layer_sizes = [64, 50, 20, 10]

    # Adds the layers to the neural network using the default bias scheme
    nn.add_layers(layer_sizes)

    # Adds the layers to the neural network
    # nn.add_layer(64, bias=True)
    # nn.add_layer(50, bias=True)
    # nn.add_layer(20, bias=True)
    # nn.add_layer(10, bias=False)

    # Trains the model
    costs = nn.train(X_train, y_train, epochs=100, learning_rate=1, batch_size=40, regularization_weight=0.00025)

    # Predicts from the test set and calculates the accuracy
    predictions = nn.predict(X_test)
    accuracy = nn.calc_accuracy(y_test)
    #nn.save("l1", min_acc=0.95)

    # Plots
    plt.plot(costs)
    plt.show()

    # Loads a pre-trained model and uses it to predict
    # nn.load("l1")
    # predictions = nn.predict(X_test)
    # accuracy = nn.calc_accuracy(y_test)




main()
