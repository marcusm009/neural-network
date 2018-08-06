from random import randint
import numpy as np
from Layer import Layer
import collections
import itertools
import pickle

#np.set_printoptions(precision=8)

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
        file = open(name + ".pckl", "rb")
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
    def train(self, X, y, learning_rate=0.1, epochs=1, grad_check=False, batch_size=10, regularization_weight=0, view=False):
        # Calculate training size (m) and the feature size (n)
        m,n = X.shape

        # Initialize list of costs for plotting
        costs = []

        # Check to make sure feature size matches up with first layer size
        if n != self.layers[0].size:
            print("Error! Dimension of feature vector (n=" + str(n) +") doesn't match dimension of input layer (n=" + str(self.layers[0].size) + ").")
            return

        for i in range(epochs):
            for batch in NeuralNetwork.iterate_batches(X, y, batch_size):
                X_batch, y_batch = batch
                activations = self.forward_prop(X_batch)
                grads = self.backprop(activations, y_batch, regularization_weight)
                self.update_theta(grads, learning_rate)

                # Append cost of this iteration to list
                costs.append(self.cost(X_batch, y_batch, regularization_weight))

            if grad_check:
                activations = self.forward_prop(X)
                grads = self.backprop(activations, y, regularization_weight)
                grads_approx = self.check_gradient(X, y, regularization_weight, (1,0,0))
                print("Grad approx: " + str(grads_approx))
                print("Your grad: " + str(grads))

            if view:
                self.display()

        return costs

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
    def backprop(self, activations, y, regularization_weight):
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
            # Trim the gradient when the next layer has bias
            if self.layers[i+1].has_bias:
                p_d_theta = p_d_theta[:,1:]
            # Add the regularization term to each layer
            p_d_theta += regularization_weight * self.layers[i].get_regularization()
            D.append(p_d_theta)
        return D

    def iterate_batches(X, y, batch_size, shuffle=True):
        # Asserts that the number of training examples matches up with its output
        assert X.shape[0] == y.shape[0]

        # Shuffle the training examples
        if shuffle:
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)

        # Loop through all training examples, with a step of batch size
        for i_start in range(0, X.shape[0] - batch_size + 1, batch_size):
            # Shuffle using the indices if they was defined or the slice function
            if shuffle:
                excerpt = indices[i_start:i_start + batch_size]
            else:
                excerpt = slice(i_start, i_start + batch_size)
            # Yield the X and y slices to form the batch
            yield X[excerpt], y[excerpt]

    # Updates thetas of all layers
    def update_theta(self, grads, learning_rate):
        for i, layer in enumerate(self.layers):
            # If i is the last layer, don't update theta
            if i == self.num_layers - 1:
                break

            layer_grad = grads[i]
            layer.theta += -1*learning_rate*layer_grad

    # TODO: Fix gradient checking algorithm
    def check_gradient(self, X, y, idx, regularization_weight, epsilon=0.0000001):
        self.layers[idx[0]].theta[idx[1],idx[2]] -= epsilon
        cost1 = self.cost(X, y, regularization_weight)

        self.layers[idx[0]].theta[idx[1],idx[2]] += 2*epsilon
        cost2 = self.cost(X, y, regularization_weight)

        self.layers[idx[0]].theta[idx[1],idx[2]] += epsilon
        grad_approx = (cost2 - cost1)/(2*epsilon)
        return grad_approx

    # A bit of a hacky way to get and set every theta in an "unrolled" fashion
    # def theta_unrolled(self):
    #     for layer in self.layers:
    #         n = layer.size
    #         if layer.has_bias:
    #             n += 1
    #         if hasattr(layer, 'theta'):
    #             for i in range(n):
    #                 for j in range(layer.next_size):
    #                     # Returns a tuple of lambda functions representing a getter and setter
    #                     yield lambda _: layer.get_theta_at_index(i, j), lambda y: layer.set_theta_at_index(i, j, y)

    # TODO: Clean up cost function
    def cost(self, X, y, regularization_weight):
        m = len(X) + 1
        cost = (-1/m)*np.sum(np.multiply(y, np.log(self.predict(X)) + np.multiply((1 - y), np.log(1 - self.predict(X)))))
        regularization = 0
        # Loop through layers to amass a regularization term
        for layer in self.layers[:-1]:
            regularization += np.sum((regularization_weight/(2*m))*(layer.get_regularization()**2))
        cost += regularization
        return cost

    def display(self):
        print("="*20)
        for layer in self.layers:
            layer.display()
