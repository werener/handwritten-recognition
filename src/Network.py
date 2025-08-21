import random
import numpy as np


class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(x, y) for x, y in zip(sizes[:-1], sizes[1:])]

        def feedforward(self, a_prev):
            for w, b in zip(self.weights, self.biases):
                a_new = sigmoid(np.dot(w, a_prev) + b)
            return a_prev

        def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
            if test_data:
                n_test = len(test_data)

            n = len(training_data)
            for j in range(epochs):
                random.shuffle(training_data)
                mini_batches = [
                    training_data[k : k + mini_batch_size]
                    for k in range(0, n, mini_batch_size)
                ]
                for mini_batch in mini_batches:
                    self.update_mini_batch(mini_batch, eta)
                if test_data:
                    print("Epoch {0}: {1} / {2}", j, self.evaluate(test_data), n_test)
                else:
                    print("Epoch {0} complete", j)

        def update_mini_batch(self, mini_batch, eta):
            nabla_b = [np.zeros(b.shape) for b in self.biases]
            nabla_w = [np.zeros(w.shape) for w in self.weights]
            for x, y in mini_batch:
                delta_nabla_b, delta_nabla_w = self.backprop(x, y)
                nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
                
            self.weights = [w-(eta/len(mini_batch))*nw
                            for w, nw in zip(self.weights, nabla_w)]
            self.biases = [b-(eta/len(mini_batch))*nb
                        for b, nb in zip(self.biases, nabla_b)]

net = Network([1, 3, 4])
print([x for x in zip([1, 2, 3, 4], [3, 4, 5, 7])])


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))
