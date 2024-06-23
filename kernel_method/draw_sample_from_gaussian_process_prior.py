import random
import sys

import matplotlib.pyplot as plt
import numpy as np


def gaussian_kernel(x_train, x_test):
    res = np.zeros(len(x_test))
    sigma = 0.5
    for i, x in enumerate(x_test):
        res[i] = np.exp(-np.sum(abs(x - x_train)) ** 2 / 2 * sigma**2)

    return res


def main():
    # Create dataset
    random.seed(314)
    x = np.linspace(-1, 1, 300)
    E = np.zeros_like(x)

    params = [
        (1, 4, 0, 0),
        (9, 4, 0, 0),
        (1, 64, 0, 0),
        (1, 0.25, 0, 0),
        (1, 4, 10, 0),
        (1, 4, 0, 5),
    ]

    fig, ax = plt.subplots(2, 3, figsize=(12, 6))
    ax = ax.ravel()
    for axi, (p0, p1, p2, p3) in zip(ax, params):
        K = (
            p0 * np.exp(-p1 / 2 * (x[:, np.newaxis] - x[np.newaxis, :]) ** 2)
            + p2
            + p3 * x[:, np.newaxis] * x[np.newaxis, :]
        )
        axi.plot(x, np.random.multivariate_normal(E, K, size=5).T)
        axi.set_title(f"({p0}, {p1}, {p2}, {p3})")

    fig.suptitle("Sample from a Gaussian process prior")


if __name__ == "__main__":
    main()
    if len(sys.argv) == 2 and sys.argv[1] == "NotShow":
        print("It shows nothing")
    else:
        plt.show()
