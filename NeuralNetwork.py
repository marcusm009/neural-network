from random import randint
import numpy as np
from Layer import Layer

class NeuralNetwork:

    def __init__(self):
        self.layers = []
        self.num_layers = 0

    def add_layer(self, size, bias=True):
        layer = Layer(self.num_layers, size, bias)

        ### Initialize theta of the last layer ###

        # Checks to see if there is a previous layer
        if self.num_layers > 0:
            # Uses information about new layer to initialize theta
            self.layers[self.num_layers - 1].initialize_theta(layer.size)

        ### Appends new layer ###
        self.layers.append(layer)
        self.num_layers += 1

    def predict(self, x):

        # Loop through subsequent layers and return output
        for layer in self.layers:
            layer.a = x
            x = layer.fire()

        # Return output of final layer
        return x

    def train(self, X, y):
        # Calculate delta of last layer
        delta = []
        prediction = self.predict(X)
        delta.append(prediction - y)

        # Calculate delta of previous layers
        l = self.num_layers - 2
        while l > 0:
            d = np.transpose(self.layers[l].theta)
            prod = np.dot(delta[len(delta) - 1], d)
            g_prime = np.multiply(self.layers[l].a, 1 - self.layers[l].a)
            delta.append(np.multiply(prod, g_prime))
            l -= 1

        delta = list(reversed(delta))

        # Calculate the derivatives of the cost function
        D = []
        for i in range(self.num_layers - 1):
            D.append(np.dot(np.transpose(self.layers[i].a), delta[i]))

        return D

    def cost(self, X, y):
        m = len(X) + 1
        cost = (-1/m)*np.sum(np.multiply(y, np.log(self.predict(X))) + np.multiply((1 - y), np.log(1 - self.predict(X))))
        return cost

    def display(self):
        print("="*20)
        for layer in self.layers:
            layer.display()

'''
    ## Experimental ##

    def add_layer(self, size, bias="True"):
        # Add the bias
        if bias:
            temp = np.concatenate((np.array([1]), np.zeros(size)))
            size += 1
        else:
            temp = np.zeros(size)

        # Expand dimensions to make dim x 1
        new_row = np.expand_dims(temp, 1)

        # Add the attribute if the nn doesn't have any layers
        if hasattr(self, 'a') == False:
            self.a = new_row
            self.max_layer_size = size
            self.num_layers = 1
            return

        # Checks for reshapes
        if size > self.max_layer_size:
            self.a = np.concatenate((self.a, np.zeros((self.num_layers, size - self.max_layer_size))))
        elif size < self.max_layer_size:
            new_row = np.concatenate((new_row, np.zeros(self.max_layer_size - size)))

        # Adds new layer and raises layer count
        self.a = np.hstack((self.a, new_row))
        self.num_layers += 1
'''
