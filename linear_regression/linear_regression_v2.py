import numpy as np
import matplotlib.pyplot as plt


# y = a + bx
def linear_regression_two_features(x, y, a=1, b=1,
                                   learning_rate=0.0001,
                                   regularization=0,
                                   max_iterations=1000):
    normalized_x = normalize(x)
    for i in range(max_iterations):
        a, b = gradient_descent(normalized_x, y, a, b, learning_rate, regularization)

    print_result(normalized_x, y, a, b)


def normalize(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)


def calculate_error(a, b, x, y):
    sample_size = np.size(x)
    loss = np.float64(0)

    for i in range(np.size(x)):
        y_pred = a + b * x[i]
        loss += ((y[i] - y_pred) / sample_size) ** 2
    return loss


def gradient_descent(x, y, a, b, learning_rate, regularization):
    sample_size = np.size(x)

    a_gradient = np.float64(0)
    b_gradient = np.float64(0)

    for i in range(sample_size):
        cost = (a + b * x[i]) - y[i]
        a_gradient += (2 / sample_size) * learning_rate * cost
        b_gradient += (2 / sample_size) * learning_rate * x[i] * cost

    updated_a = a - a_gradient
    updated_b = b * (1 - ((learning_rate * regularization)/sample_size)) - b_gradient

    return updated_a, updated_b


def print_result(x, y, a, b):
    print("Estimated coefficients :\na = {}  \nb = {}".format(a, b))

    plt.scatter(x, y, color="g",
                marker="o", s=30)
    y_pred = a + b * x
    plt.plot(x, y_pred, color="b")
    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()
