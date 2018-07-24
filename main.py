import numpy as np
from Layer import Layer
from NeuralNetwork import NeuralNetwork

def main():
    input = np.array([1,1,1])
    y = np.array([1,1,1])

    #n = Neuron(2,2)
    #n.initializeTheta(5)
    #n.input(X)
    #n.display()
    nn = NeuralNetwork()

    nn.add_intercept_layer(3)
    nn.add_intercept_layer(4)
    nn.add_no_intercept_layer(3)
    nn.display()

    error = nn.train(input, y)
    nn.display()
    print(error)

    #nn.addInput([2,2])
    #nn.display()

    # initialize neural netowrk (input, hidden, output)
    #nn = NeuralNetwork(5)
    # read data




main()
