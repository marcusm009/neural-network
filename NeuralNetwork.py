from random import randint
import numpy as np
from Layer import Layer
import collections
import itertools
import pickle

class NeuralNetwork:

    def __init__(self):
        self.layers = []
        self.num_layers = 0
        self.accuracy = 0

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

    # A wrapper function for forward_prop that returns only the activations
    # from the last layer
    def predict(self, x):
        # Calculate training size (m) and the feature size (n) (Unused)
        if x.ndim == 1:
            n = x.size
        else:
            n,m = x.shape

        return self.forward_prop(x)[-1]

    # Calculates the accuracy of the test set
    def calc_accuracy(self, y):
        predictions = self.layers[-1].a
        right = 0
        for i, prediction in enumerate(predictions.argmax(axis=1)):
            if prediction == y.argmax(axis=1)[i]:
                right += 1
        self.accuracy = right/predictions.argmax(axis=1).size
        print("Accuracy: " + str(self.accuracy))

    # Loads from a file
    def load(self, name):
        file = open(name + ".pckl", 'rb')
        new_layers = pickle.load(file)
        file.close()
        self.layers = new_layers
        self.num_layers = len(self.layers)
        print("Model loaded!")

    # Saves to a file
    def save(self, name, min_acc=0.9, check=True):
        if self.accuracy > min_acc or check == False:
            file = open(name + ".pckl", "wb")
            pickle.dump(self.layers, file)
            file.close()
            print("Model saved!")
        else:
            print("Model did not meet minimum accuracy of " + str(min_acc) + ". Accuracy was " + str(self.accuracy) + ".")
            print("If you wish to save anyways, pass in a lower 'min_acc' value or pass in 'check=False'.")

    # The main function for training the model
    def train(self, X, y, learning_rate=0.1, iterations=1000, grad_check=False):
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

    # Function that performs forward propagation
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

    # Function that performs backwards propagation
    def backprop(self, activations, y):
        # Calculate delta of last layer
        delta = []
        # Calculate the amount of training examples
        m = activations[0].shape[0]
        prediction = activations[-1]
        delta.append(prediction - y)

        # Calculate delta of previous layers
        l = self.num_layers - 2

        while l > 0:
            # Trims delta if the next layer has bias
            # Potential cause of layer BUG
            if self.layers[l+1].has_bias:
                delta_prev = delta[-1][:,1:]
            else:
                delta_prev = delta[-1]

            # Algorithm for forward propagation
            prod = np.dot(delta_prev, self.layers[l].theta.T)
            g_prime = np.multiply(self.layers[l].a, 1 - self.layers[l].a)
            delta.append(np.multiply(prod, g_prime))
            l -= 1

        delta = list(reversed(delta))

        # Calculate the partial derivatives with respect to theta
        D = []
        for i in range(self.num_layers - 1):
            p_d_theta = self.layers[i].a.T.dot(delta[i]) / m
            print("partial " + str(i) + ": " + str(p_d_theta.shape))
            D.append(p_d_theta)
        return D

    # Updates thetas of all layers
    def update_theta(self, grads, learning_rate):
        for i, layer in enumerate(self.layers):
            # If i is the last layer, don't update theta
            if i == self.num_layers - 1:
                break

            # Update theta according to layer grad
            # Potential cause of layer BUG
            layer_grad = grads[i]
            if self.layers[i+1].has_bias:
                layer_grad = layer_grad[:,1:]
            layer.theta += -1*learning_rate*layer_grad

    # TODO: Fix gradient checking algorithm
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

    # TODO: Fix cost function
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
