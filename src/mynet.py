import numpy as np
import random
from collections import deque

import load_mnist_data


class Network:
    def __init__(self, sizes):
        self.L = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [
            np.random.randn(y, x) / np.sqrt(x)
            for x, y in zip(self.sizes[:-1], self.sizes[1:])
        ]

    def feedforward(self, a):
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(w @ a + b)
        return a

    def learn(self, settings: dict):
        training_data = settings["train_data"]
        mini_batch_size = settings["mini_batch_size"]
        learn_rate = settings["learn_rate"]
        lambda_ = settings["lambda"]

        if "epochs" in settings:
            epochs = settings["epochs"] or 10
        else:
            epochs = 10

        if "no_improvement_in_n" in settings:
            no_improvement_in_n = settings["no_improvement_in_n"] or epochs
        else:
            no_improvement_in_n = epochs

        if "validation_data" in settings:
            validation_data = settings["validation_data"] or None
        else:
            validation_data = None

        if "test_data" in settings:
            test_data = settings["test_data"] or None
        else:
            test_data = None

        self.gradient_descent(
            training_data,
            mini_batch_size,
            learn_rate,
            lambda_,
            epochs,
            no_improvement_in_n,
            validation_data,
            test_data,
        )

    @staticmethod
    def is_stagnating(dq: deque):
        improvement_percent_leeway = 0.1
        dq = list(dq)
        min_acc = min(dq[1:])
        # print(f"{list(map(int, dq))} -> {(dq[0] - min_acc) / min_acc * 100}%")
        return (dq[0] - min_acc) / min_acc * 100 <= improvement_percent_leeway

    def gradient_descent(
        self,
        training_data,
        mini_batch_size,
        learn_rate,
        lambda_,
        epochs,
        no_improvement_in_n,
        validation_data,
        test_data,
    ):
        n = len(training_data)
        if validation_data:
            accuracies = deque([0] * no_improvement_in_n)

        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[i : i + mini_batch_size]
                for i in range(0, n, mini_batch_size)
            ]
            for batch in mini_batches:
                self.update_with_mini_batch(
                    batch, learn_rate, lambda_, len(training_data)
                )

            if validation_data:
                accuracy_eval = self.accuracy(validation_data)
                accuracies.pop()
                accuracies.appendleft(accuracy_eval)
                print(
                    f"Epoch {epoch + 1}:\n   valid:  {accuracy_eval} / {len(validation_data)}"
                )
                if epoch > no_improvement_in_n - 2 and self.is_stagnating(accuracies):
                    
                    if test_data:
                        print(
                            f"   test:  {self.accuracy(test_data)} / {len(test_data)}"
                        )
                    print(
                        f"There has been no improvements last {no_improvement_in_n} epochs. Terminating"
                    )
                    break

            if test_data:
                print(f"   test:  {self.accuracy(test_data)} / {len(test_data)}")

    def update_with_mini_batch(self, batch, eta, lambda_, train_len):
        gradient_b = [np.zeros(b.shape) for b in self.biases]
        gradient_w = [np.zeros(w.shape) for w in self.weights]
        for input_layer, expected_result in batch:
            change_grad_b, change_grad_w = self.backpropagation(
                input_layer, expected_result
            )
            gradient_b = [
                prev + change for prev, change in zip(gradient_b, change_grad_b)
            ]
            gradient_w = [
                prev + change for prev, change in zip(gradient_w, change_grad_w)
            ]

        n = len(batch)
        self.biases = [b - (eta / n) * gb for b, gb in zip(self.biases, gradient_b)]
        self.weights = [
            (1 - eta * (lambda_ / train_len)) * w - (eta / n) * gw
            for w, gw in zip(self.weights, gradient_w)
        ]

    def backpropagation(self, x, y):
        cost_gradient_b = [np.zeros(b.shape) for b in self.biases]
        cost_gradient_w = [np.zeros(w.shape) for w in self.weights]
        zs = []
        activations = [x]
        for b_layer, w_layer in zip(self.biases, self.weights):
            z = w_layer @ activations[-1] + b_layer
            zs.append(z)
            activations.append(sigmoid(z))

        # в уме держим, что dC/da^L = (a-y)/sigmoid`(z), а delta = dC/da^L * sigmoid`(z)
        delta = dC_daL(activations[-1], y)
        cost_gradient_b[-1] = delta
        cost_gradient_w[-1] = delta @ activations[-2].T

        for layer in range(2, self.L):
            delta = (self.weights[-layer + 1].T @ delta) * sigmoid_der(zs[-layer])
            cost_gradient_b[-layer] = delta
            cost_gradient_w[-layer] = delta @ activations[-layer - 1].T

        return (cost_gradient_b, cost_gradient_w)

    def accuracy(self, data, is_train_data=False):
        if is_train_data:
            results = [
                (np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in data
            ]
        else:
            results = [(np.argmax(self.feedforward(x)), y) for (x, y) in data]

        return sum(x == y for x, y in results)


def dC_daL(output_layer_activations, desired_output_activations):
    return output_layer_activations - desired_output_activations


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_der(z):
    return sigmoid(z) * (1 - sigmoid(z))


if __name__ == "__main__":
    train, valid, test = load_mnist_data.load_data_wrapper()
    net = Network([784, 50, 10])
    settings = {
        "train_data": train,
        "mini_batch_size": 20,
        "learn_rate": 0.1,
        "lambda": 5.0,
        "epochs": 80,
        "no_improvement_in_n": 5,
        "validation_data": valid,
        "test_data": test,
    }
    net.learn(settings)
