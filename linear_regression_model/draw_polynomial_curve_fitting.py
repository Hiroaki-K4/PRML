import random

import matplotlib.pyplot as plt
import numpy as np


def update_weights(x, y, dim):
    A = np.zeros((dim + 1, dim + 1))
    T = np.zeros(dim + 1)
    for i in range(dim + 1):
        for j in range(dim + 1):
            multi_x = np.array(x) ** (i + j)
            A[i, j] = sum(multi_x)
        T[i] = np.dot(np.array(x) ** i, np.array(y))

    return np.dot(np.linalg.inv(A), T)


def predict(W, x):
    pred = np.zeros(x.shape)
    for i in range(len(W)):
        pred += np.dot(W[i], x**i)

    return pred


def main():
    # Create dataset
    random.seed(0)
    x = np.linspace(0, 2 * np.pi, 500)
    y = np.sin(x)
    N = 30
    nums = random.sample(range(x.shape[0]), k=N)
    noise_x = []
    noise_y = []
    for idx in nums:
        random_x = x[idx]
        random_y = y[idx] + np.random.normal(0, 0.2)
        noise_x.append(random_x)
        noise_y.append(random_y)

    degree = 10
    W = update_weights(np.array(noise_x), np.array(noise_y), degree)
    pred = predict(W, np.array(x))

    plt.plot(x, y, label="True")
    plt.plot(x, pred, label="Prediction")
    plt.scatter(noise_x, noise_y, label="Noise points")
    plt.legend()
    title = "Polynomial curve fitting(degree={0},N={1})".format(str(degree), str(N))
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    main()
