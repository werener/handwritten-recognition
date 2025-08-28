if __name__ == "__main__":
    import load_mnist_data, Network, NetworkMatrixBased

    training_data, validation_data, test_data = load_mnist_data.load_data_wrapper()
    net = NetworkMatrixBased.Network([784, 30, 10])
    net.gradient_descend(training_data, 30, 10, 3.0, test_data=test_data)
