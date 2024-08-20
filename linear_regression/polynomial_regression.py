import numpy as np
import matplotlib.pyplot as plt


def polynomial_regression(x, y, learning_rate=0.0001,
                          initial_parameters=None,
                          normalization=True,
                          max_iterations=100,
                          degree=2):
    if normalization:
        x = normalize(x)

    if initial_parameters is None or len(initial_parameters) != degree + 1:
        parameters = np.zeros(degree + 1)
    else:
        parameters = np.array(initial_parameters)

    for i in range(0, max_iterations):
        parameters = gradient_descent(x, y, parameters, learning_rate)

    print_result(x, y, parameters)


def normalize(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)


def gradient_descent(x, y, parameters, learning_rate):
    sample_size = np.size(x)
    gradient_parameters = np.zeros_like(parameters)

    for i in range(sample_size):
        for j in range(len(parameters)):
            gradient_parameters[j] += (2 / sample_size) * learning_rate * (np.polyval(parameters, x[i]) - y[i]) * (
                        x[i] ** j)

    updated_parameters = parameters - gradient_parameters
    return updated_parameters


def print_result(x, y, parameters):
    x_plot = np.linspace(np.min(x), np.max(x), 100)
    y_plot = np.polyval(parameters, x_plot)
    plt.plot(x_plot, y_plot, color="b")
    plt.scatter(x, y, color="g")
    plt.show()
