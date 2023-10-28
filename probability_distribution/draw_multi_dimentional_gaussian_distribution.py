import sys

import matplotlib.pyplot as plt
import numpy as np


def calculate_multi_dimentional_gaussian_distribution(mu, cov, vals):
    if mu.shape[0] != cov.shape[0] or mu.shape[0] != cov.shape[1]:
        raise ValueError("Shape of mu and cov are not same.")

    density = np.empty((vals.shape[0], vals.shape[1]))
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            density[i][j] = (
                (1 / (2 * np.pi) ** (mu.shape[0] / 2))
                * (1 / np.linalg.det(cov) ** 0.5)
                * np.exp(
                    -1
                    / 2
                    * np.dot(
                        np.dot(np.transpose(vals[i][j] - mu), np.linalg.inv(cov)),
                        (vals[i][j] - mu),
                    )
                )
            )

    return density


def main():
    N = 100
    X = np.linspace(-3, 3, N)
    Y = np.linspace(-3, 3, N)
    X, Y = np.meshgrid(X, Y)

    mu = np.array([0, 0])
    cov = np.array([[1.0, 0.5], [0.5, 1.5]])

    vals = np.empty(X.shape + (2,))
    vals[:, :, 0] = X
    vals[:, :, 1] = Y

    density = calculate_multi_dimentional_gaussian_distribution(mu, cov, vals)

    fig = plt.figure()
    fig.suptitle(t="Multi dimentional gaussian distribution", fontsize=20)
    ax = fig.add_subplot(projection="3d")
    ax.set_xlabel(xlabel="Dimention 1")
    ax.set_ylabel(ylabel="Dimention 2")
    ax.set_zlabel(zlabel="Density")
    param_text = "$\\mu=({0},{1})$, cov={2}".format(mu[0], mu[1], cov)
    ax.set_title(label=param_text, loc="left")
    ax.plot_surface(X, Y, density, cmap="viridis")


if __name__ == "__main__":
    main()
    if len(sys.argv) == 2 and sys.argv[1] == "NotShow":
        print("It shows nothing")
    else:
        plt.show()
