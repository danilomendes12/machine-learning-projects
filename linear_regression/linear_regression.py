import numpy as np
import matplotlib.pyplot as plt


# y = a + bx
# this script is incorrect, the v2 should be used instead
def linear_regression_two_features(x, y,
                                   starting_a_value=1,
                                   starting_b_value=1,
                                   max_iterations=1000,
                                   learning_rate=0.0001,
                                   convergence_criteria=0.001,
                                   normalization=True):
    sample_size = np.size(x)

    a = starting_a_value
    b = starting_b_value

    previous_cost = None

    for i in range(0, max_iterations):
        (a, b) = update_coefficients(x, y, a, b, learning_rate, sample_size)

        cost = get_cost(x, y, a, b)

        if previous_cost is not None and previous_cost < cost:
            break

        if previous_cost is not None and abs(previous_cost - cost) < convergence_criteria:
            break

        previous_cost = cost

    return a, b


def print_result(x, y, b):
    print("Estimated coefficients:\na = {}  \nb = {}".format(b[0], b[1]))

    plt.scatter(x, y, color="g",
                marker="o", s=30)
    y_pred = b[0] + b[1] * x
    plt.plot(x, y_pred, color="b")
    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()


def normalize(x):
    x_max_value = np.max(x)
    x_min_value = np.min(x)

    return (x - x_min_value) / (x_max_value - x_min_value)


def update_coefficients(x, y, a, b, learning_rate, sample_size):
    new_a = a - (learning_rate * (1 / sample_size) * get_cost_for_a(x, y, a, b))
    b = b - (learning_rate * (1 / sample_size) * get_cost_for_b(x, y, a, b))
    return new_a, b


def get_cost(x, y, a, b):
    cost: int = 0

    for (x_value, y_value) in zip(x, y):
        h = a + b * x_value
        cost += (h - y_value) ** 2
    # print(f"cost: {cost}")
    return cost


def get_cost_for_a(x, y, a, b):
    cost: int = 0

    for (x_value, y_value) in zip(x, y):
        h = a + b * x_value
        cost += h - y_value
    return cost


def get_cost_for_b(x, y, a, b):
    cost: int = 0

    for (x_value, y_value) in zip(x, y):
        h = a + b * x_value
        cost += (h - y_value) * x_value
    return cost
