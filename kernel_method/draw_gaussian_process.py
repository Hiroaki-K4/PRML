import random
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def calculate_kernel(x_train, x):
    params = [1, 4, 0, 0]
    k = (
        params[0] * np.exp(-params[1] / 2 * (x[:, np.newaxis] - x[np.newaxis, :]) ** 2)
        + params[2]
        + params[3] * x[:, np.newaxis] * x[np.newaxis, :]
    )
    print(k)
    input()


def predict_by_gaussian_process(x_train, y_train, x_test):
    preds = np.zeros(len(x_test))
    # TODO: Calculate k and C
    print(x_train)
    print(x_train.shape)
    for i, x in enumerate(x_test):
        calculate_kernel(x_train, x)
        probs = norm.pdf(x - x_train, loc=0, scale=0.2)
        preds[i] = np.sum(probs * y_train) / np.sum(probs)

    return preds


def main():
    # Create dataset
    random.seed(0)
    x = np.linspace(0, 2 * np.pi, 500)
    y = np.sin(x)
    N = 25
    nums = random.sample(range(x.shape[0]), k=N)
    noise_x = []
    noise_y = []
    for idx in nums:
        random_x = x[idx]
        random_y = y[idx] + np.random.normal(0, 0.2)
        noise_x.append(random_x)
        noise_y.append(random_y)

    preds = predict_by_gaussian_process(
        np.array(noise_x), np.array(noise_y), np.array(x)
    )

    plt.plot(x, y, label="True", c="green")
    plt.plot(x, preds, label="Prediction", c="red")
    plt.scatter(noise_x, noise_y, label="inputs")
    plt.legend()
    title = "Nadaraya-Watson model(N={0})".format(str(N))
    plt.title(title)


if __name__ == "__main__":
    main()
    if len(sys.argv) == 2 and sys.argv[1] == "NotShow":
        print("It shows nothing")
    else:
        plt.show()
