import numpy as np

np.set_printoptions(precision=3)

class Layer:

    def __init__(self, layer_number, size):
        self.layer_number = layer_number
        self.a = np.zeros(size)

    def initialize_theta(self, size):
        self.theta = np.random.rand(self.a.size, size)

    def fire(self):
        if hasattr(self, 'theta'):
            # Reshape a
            print(self.a.shape)
            print(self.theta.shape)
            #return self.a
            return self.sigmoid(np.dot(self.a.reshape(1,self.a.size), self.theta))
        else:
            return self.a

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def display(self):
        print("Layer " + str(self.layer_number) + ":")
        print("Activations: " + str(self.a))
        if hasattr(self, 'theta'):
            print("Theta:\n" + str(self.theta))
        print("="*20)
