import random
import sys

import matplotlib.pyplot as plt
import numpy as np


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
    diago_elems = np.full(x_train.shape[0], std**2)
    C_var = np.diag(diago_elems)

    params = [1, 4, 0, 0]
    mean = np.zeros(len(x_test))
    var = np.zeros(len(x_test))
    for i in range(x_test.shape[0]):
        k = calculate_kernel(x_train, x_test[i], params)
        C = calculate_kernel_for_cov(x_train, params) + C_var
        c = calculate_kernel(x_test[i], x_test[i], params) + std
        mean[i] = np.dot(np.dot(k.T, np.linalg.inv(C)), y_train)
        var[i] = c - np.dot(np.dot(k.T, np.linalg.inv(C)), k)

    return mean, var


def main():
    # Create dataset
    random.seed(0)
    x = np.linspace(0, 2 * np.pi, 300)
    y = np.sin(x)
    N = 50
    nums = random.sample(range(int(x.shape[0] - x.shape[0] / 3)), k=N)
    noise_x = []
    noise_y = []
    std = 0.1
    for idx in nums:
        random_x = x[idx]
        noise = np.random.normal(0, std)
        random_y = y[idx] + noise
        noise_x.append(random_x)
        noise_y.append(random_y)

    mean, var = predict_by_gaussian_process(
        np.array(noise_x), np.array(noise_y), np.array(x), std
    )
    max_y_list = []
    min_y_list = []
    for i in range(var.shape[0]):
        max_y = mean[i] + np.sqrt(abs(var[i])) * 2
        min_y = mean[i] - np.sqrt(abs(var[i])) * 2
        max_y_list.append(max_y)
        min_y_list.append(min_y)

    plt.plot(x, y, label="True", c="green")
    plt.plot(x, mean, label="Prediction", c="red")
    plt.fill_between(
        x, min_y_list, max_y_list, label=r"std $\times 2$", color="orange", alpha=0.8
    )
    plt.scatter(noise_x, noise_y, label="inputs")
    plt.legend()
    title = "Regression by Gaussian process(N={0})".format(str(N))
    plt.title(title)


if __name__ == "__main__":
    main()
    if len(sys.argv) == 2 and sys.argv[1] == "NotShow":
        print("It shows nothing")
    else:
        plt.show()
