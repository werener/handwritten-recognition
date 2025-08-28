import random
import numpy as np


class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def learn(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        self.gradient_descend(training_data, epochs, mini_batch_size, eta, test_data)

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def gradient_descend(self, training_data, epochs, mini_batch_size, eta, test_data):
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
                print(f"Epoch {j}: {self.evaluate(test_data)} / {n_test}")
            else:
                print(f"Epoch {j} complete")

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        x_matrix_0 = [x for x, y in mini_batch]
        y_matrix_0 = [y for x, y in mini_batch]
        x_matrix = np.concatenate(x_matrix_0, axis=1)
        y_matrix = np.concatenate(y_matrix_0, axis=1)

        nabla_b, nabla_w = self.backprop(x_matrix, y_matrix)

        self.weights = [
            w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)
        ]
        self.biases = [
            b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)
        ]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = np.reshape([np.sum(nb) for nb in delta], [delta.shape[0], 1])
        for _d, _a in zip(delta.transpose(), activations[-2].transpose()):
            _d = np.reshape(_d, [len(_d), 1])
            _a = np.reshape(_a, [len(_a), 1])
            nabla_w[-1] += np.dot(_d, _a.transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = np.reshape([np.sum(nb) for nb in delta], [delta.shape[0], 1])
            for _d, _a in zip(delta.transpose(), activations[-l - 1].transpose()):
                _d = np.reshape(_d, [len(_d), 1])
                _a = np.reshape(_a, [len(_a), 1])
                nabla_w[-l] += np.dot(_d, _a.transpose())
        return nabla_b, nabla_w

    def cost_derivative(self, output_activations, y):
        return output_activations - y

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))
