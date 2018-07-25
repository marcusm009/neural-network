import numpy as np

np.set_printoptions(precision=3)

class Layer:

    def __init__(self, layer_number, size, bias=True):
        self.layer_number = layer_number
        self.has_bias = bias
        self.size = size # The size of the non-bias neurons
        self.a = np.zeros(size)

    def initialize_theta(self, size):
        if self.has_bias:
            self.theta = np.random.rand(self.size + 1, size)
        else:
            self.theta = np.random.rand(self.size, size)

    def fire(self):
        #print("Layer " + str(self.layer_number) + ": Fired!")

        # Fires normally by returning g(theta*a')
        if hasattr(self, 'theta'):
            # print(self.a.shape)
            # print(self.theta.shape)
            return self.sigmoid(np.dot(self.a, self.theta))
        # Fires abnormally by just returning a (usually happens on last layer)
        else:
            return self.a


    @property
    def a(self):
        return self.__a

    @a.setter
    def a(self, a):
        # m is the number of training examples
        m = a.size//self.size
        a = np.reshape(a,(m, self.size))
        if self.has_bias:
            # Appends a 1 to the beginning to account for bias
            a = np.concatenate((np.ones((m,1)),a),axis=1)
        # Transforms to column vector
        self.__a = a#.reshape(len(a),1)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def display(self):
        print("Layer " + str(self.layer_number) + ":")
        print("Activations: ")
        print(self.a)
        if hasattr(self, 'theta'):
            print("Theta:\n" + str(self.theta))
        print("="*20)
