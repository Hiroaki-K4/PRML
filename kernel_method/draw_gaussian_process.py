import random
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def calculate_kernel(x_0, x_1, params):
    k = (
        params[0] * np.exp(-params[1] / 2 * (x_0 - x_1) ** 2)
        + params[2]
        + params[3] * x_0.T * x_1
    )

    return k


def calculate_kernel_for_cov(x, params):
    C = (
        params[0] * np.exp(-params[1] / 2 * (x[:, np.newaxis] - x[np.newaxis, :]) ** 2)
        + params[2]
        + params[3] * x[:, np.newaxis] * x[np.newaxis, :]
    )

    return C


def predict_by_gaussian_process(x_train, y_train, x_test, std):
    value = std**2
    diago_elems = np.full(x_train.shape[0], std**2)
    C_var = np.diag(diago_elems)

    params = [1, 4, 0, 0]
    mean = np.zeros(len(x_test))
    var = np.zeros(len(x_test))
    beta = 1 / std
    for i in range(x_test.shape[0]):
        k = calculate_kernel(x_train, x_test[i], params)
        C = calculate_kernel_for_cov(x_train, params) + C_var
        c = calculate_kernel(x_test[i], x_test[i], params) + beta
        mean[i] = np.dot(np.dot(k.T, np.linalg.inv(C)), y_train)
        var[i] = c - np.dot(np.dot(k.T, np.linalg.inv(C)), k)

    return mean, var


def main():
    # Create dataset
    random.seed(0)
    x = np.linspace(0, 2 * np.pi, 300)
    y = np.sin(x)
    N = 20
    nums = random.sample(range(x.shape[0]), k=N)
    nums.sort()
    noise_x = []
    noise_y = []
    std = 0.1
    for idx in nums:
        random_x = x[idx]
        noise = np.random.normal(0, std)
        random_y = y[idx] + noise
        noise_x.append(random_x)
        noise_y.append(random_y)

    del_num = 8
    noise_x = noise_x[: N - del_num]
    noise_y = noise_y[: N - del_num]

    mean, var = predict_by_gaussian_process(
        np.array(noise_x), np.array(noise_y), np.array(x), std
    )

    # TODO: Plot variance
    plt.plot(x, y, label="True", c="green")
    plt.plot(x, mean, label="Prediction", c="red")
    plt.scatter(noise_x, noise_y, label="inputs")
    plt.legend()
    title = "Regression by Gaussian process(N={0})".format(str(N - del_num))
    plt.title(title)


if __name__ == "__main__":
    main()
    if len(sys.argv) == 2 and sys.argv[1] == "NotShow":
        print("It shows nothing")
    else:
        plt.show()
