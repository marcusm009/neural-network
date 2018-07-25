import numpy as np
from Layer import Layer
from NeuralNetwork import NeuralNetwork

def main():
    X = np.random.random([5,1])
    y = np.array([[0.872, 0.763, 0.783],
 [0.857, 0.756, 0.776],
 [0.85,  0.75,  0.77 ],
 [0.871, 0.761, 0.782],
 [0.845, 0.743, 0.765]]
)

    Theta0 = np.array([[0.1, 0.2]])
    Theta1 = np.array([[0.1, 0.2, 0.3],
                       [0.4, 0.5, 0.6]])



    #print(X[0])
    #print(y[0])

    #new_arr = np.array([1])
    #new_arr.reshape()
    #new_array.shape

    #n = Neuron(2,2)
    #n.initializeTheta(5)
    #n.input(X)
    #n.display()
    nn = NeuralNetwork()

    nn.add_layer(1)
    nn.add_layer(2)
    nn.add_layer(3, bias=False)

    nn.display()

    #nn.layers[0].theta = Theta0
    #nn.layers[1].theta = Theta1

    #prediction = nn.predict(X)
    #nn.display()

    #error = nn.train(X[0], y[0])
    cost = nn.cost(X, y)
    nn.display()

    #print(error)
    print(cost)
    #print(prediction)

    # initialize neural netowrk (input, hidden, output)
    #nn = NeuralNetwork(5)
    # read data




main()
