from random import randint
import numpy as np
from Layer import Layer
import collections
import itertools

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
        # Calculate training size (m) and the feature size (n)
        if x.ndim == 1:
            n = x.size
        else:
            n,m = x.shape

        # Check to make sure feature size matches up with first layer size
        # if n != self.layers[0].size:
        #     print("Error! Dimension of feature vector (n=" + str(n) +") doesn't match dimension of input layer (n=" + str(self.layers[0].size) + ").")
        #     return

        return self.forward_prop(x)[-1]

    def train(self, X, y, learning_rate=0.1, iterations=10000, grad_check=False):
        # Calculate training size (m) and the feature size (n)
        m,n = X.shape

        # Check to make sure feature size matches up with first layer size
        if n != self.layers[0].size:
            print("Error! Dimension of feature vector (n=" + str(n) +") doesn't match dimension of input layer (n=" + str(self.layers[0].size) + ").")
            return

        for i in range(iterations):
            self.display()
            activations = self.forward_prop(X)
            grads = self.backprop(activations, y)
            print("Iteration: " + str(i))
            print("Cost: " + str(self.cost(X, y)))
            #print("Calculated grads: " + str(grads))
            if grad_check:
                grad_approx = self.check_gradient(X, y)
                print("Grad approx: " + str(grad_approx))
            self.update_theta(grads, learning_rate)

    def forward_prop(self, x):
        activations = []
        activations.append(x)
        # Loop through subsequent layers and return output
        for layer in self.layers:
            layer.a = activations[-1]
            #self.a.append(layer.a)
            activations.append(layer.fire())

        # Return the activations
        return activations

    def backprop(self, activations, y):
        # Calculate delta of last layer
        delta = []
        m,n = activations[0].shape
        prediction = activations[-1]
        delta.append(prediction - y)

        # Calculate delta of previous layers
        l = self.num_layers - 2

        #for layer in reversed(self.layers[:self.num_layers - 2]):
            #delta.append(np.multiply(delta[-1].dot(layer.theta.T),sigmoid_der(layer)))
        while l > 0:
            d = np.transpose(self.layers[l].theta)
            prod = np.dot(delta[-1], d)
            g_prime = np.multiply(self.layers[l].a, 1 - self.layers[l].a)
            delta.append(np.multiply(prod, g_prime))
            l -= 1

        delta = list(reversed(delta))

        # Calculate the derivatives of the cost function
        D = []
        for i in range(self.num_layers - 1):
            D.append(self.layers[i].a.T.dot(delta[i]) / m)
        return D

    def update_theta(self, grads, learning_rate):
        for i, layer in enumerate(self.layers):
            # If i is the last layer, don't update theta
            if i == self.num_layers - 1:
                break
            layer_grad = grads[i]
            if self.layers[i+1].has_bias:
                layer_grad = layer_grad[:,1:]
            layer.theta += -1*learning_rate*layer_grad

    def check_gradient(self, X, y, epsilon=0.0001):
        l = self.num_layers - 1
        for layer in self.layers[:l]:
            layer.theta -= epsilon
        cost1 = self.J(X, y)
        for layer in self.layers[:l]:
            layer.theta += 2*epsilon
        cost2 = self.J(X,y)
        for layer in self.layers[:l]:
            layer.theta += epsilon
        grad_approx = (cost2 - cost1)/(2*epsilon)
        return grad_approx

    def cost(self, X, y):
        return np.sum(self.J(X,y))

    def J(self, X, y):
        m = len(X) + 1
        J = (-1/m)*np.multiply(y, np.log(self.predict(X)) + np.multiply((1 - y), np.log(1 - self.predict(X))))
        return J

    def display(self):
        print("="*20)
        for layer in self.layers:
            layer.display()
