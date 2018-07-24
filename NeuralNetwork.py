from random import randint
import numpy as np
from Layer import Layer

class NeuralNetwork:

    def __init__(self):
        self.layers = []

    def add_intercept_layer(self, size):
        new_layer = Layer(len(self.layers), size + 1)
        # Make the activation for the first neuron 1 to act as intercept
        new_layer.a[0] = 1
        new_layer.is_intercept_layer = True
        self._add(new_layer)

    def add_no_intercept_layer(self, size):
        new_layer = Layer(len(self.layers), size)
        new_layer.is_intercept_layer = False
        self._add(new_layer)

    def _add(self, layer):
        ### Initialize theta of the last layer ###

        # Checks to see if there is a previous layer
        if len(self.layers) > 0:
            # Uses information about new layer to initialize theta
            self.layers[len(self.layers) - 1].initialize_theta(layer.a.size)

        ### Appends new layer ###
        self.layers.append(layer)


    def predict(self, X):
        # If the first layer has an intercept, append a 1 to the input
        if self.layers[0].is_intercept_layer == True:
            X = np.concatenate((np.array([1]), X))

        # Transpose X to make it compatible with neural network
        #activation = np.transpose(X)
        activation = X

        # Loop through subsequent layers and return output
        for layer in self.layers:
            layer.a = activation
            activation = layer.fire()

        # Return output of final layer
        return activation

    def train(self, X, y):
        #error = []
        prediction = self.predict(X)
        error = prediction - y
        return error

    def display(self):
        print("="*20)
        for layer in self.layers:
            layer.display()
