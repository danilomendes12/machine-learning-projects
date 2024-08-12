import numpy as np
import matplotlib.pyplot as plt


def get_precision_and_recall(x, y, a, b):
    # todo: implement
    pass


def binomial_logistic_regression(x, y):
    sample_size = np.size(x)

    a = 1.5
    b = 1.5

    previous_cost = None

    for i in range(0, 10000):
        (a, b) = update_coefficients(x, y, a, b, 0.001, sample_size)

        cost = get_cost(x, y, a, b)

        if previous_cost is not None and previous_cost < cost:
            break

        if previous_cost is not None and abs(previous_cost - cost) < 0.001:
            break

        previous_cost = cost

    return a, b


def get_cost(x, y, a, b):
    cost = 0

    for (x_value, y_value) in zip(x, y):
        h = a * x_value + b
        cost += y_value * np.log(h) + (1 - y_value) * np.log(1 - h)
    print(f"cost: {cost}")
    return cost


def update_coefficients(x, y, a, b, learning_rate, sample_size):
    new_a = a - (learning_rate * (1 / sample_size) * get_cost_for_a(x, y, a, b))
    b = b - (learning_rate * (1 / sample_size) * get_cost_for_b(x, y, a, b))
    a = new_a
    return a, b


def get_cost_for_a(x, y, a, b):
    cost = 0

    for (x_value, y_value) in zip(x, y):
        h = a * x_value + b
        cost += (h - y_value) * x_value
    return cost


def get_cost_for_b(x, y, a, b):
    cost = 0

    for (x_value, y_value) in zip(x, y):
        h = a * x_value + b
        cost += (h - y_value)
    return cost


def predict(x, a, b):
    return 1 / (1 + np.exp(-(a * x + b)))


def main():
    x = np.random.randint(0, 41, size=40)
    y = [1 if i > 20 else 0 for i in x]

    ##todo - add logistic regression and precision and recall calculation
    a,b = binomial_logistic_regression(x, y)

    plt.scatter(x, y, color="g",
                marker="o", s=30)
    plt.plot(x, predict(x, a, b), color="b")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


if __name__ == "__main__":
    main()

#%%
