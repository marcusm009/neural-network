import numpy as np

class Layer:

    def __init__(self, layer_number, size, bias=True):
        self.layer_number = layer_number
        self.has_bias = bias
        self.size = size # The size of the non-bias neurons
        self.a = np.zeros(size)

    def __iter__(self):
        n = self.size
        if self.has_bias:
            n += 1
        if hasattr(self, 'theta'):
            for i in range(n):
                for j in range(self.next_size):
                    yield self.theta[i,j]
        else:
            yield None
            return

    def initialize_theta(self, size):
        self.next_size = size
        if self.has_bias:
            self.theta = np.random.rand(self.size + 1, self.next_size)
        else:
            self.theta = np.random.rand(self.size, self.next_size)

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

    def input_delta(self, output_delta):
        return np.multiply(self.sigmoid_der(self.a),output_delta)

    def grad(self, X, output_delta):
        JW = np.dot(X.T, output_delta)
        Jb = np.sum(output_delta, axis=0)
        return [g for g in itertools.chain(np.nditer(JW), np.nditer(Jb))]

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
        self.__a = a

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_der(self):
        return np.multiply(self.a, (1-self.a))

    def display(self):
        print("Layer " + str(self.layer_number) + ":")
        print("Activations: ")
        print(self.a)
        if hasattr(self, 'theta'):
            print("Theta:\n" + str(self.theta))
        print("="*20)
