import math
import sys

import matplotlib.pyplot as plt
import numpy as np

import draw_gaussian_distribution


def calculate_multivariate_gaussian_distribution(mu, cov, vals):
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


def calculate_conditional_gaussian_distribution(mu, cov, y):
    pre_mat = np.linalg.inv(cov)
    var_a = pre_mat[0, 0] ** (-1)
    mu_a = mu[0] - pre_mat[0, 0] ** (-1) * pre_mat[0, 1] * (y - mu[1])

    return mu_a, var_a


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

    density = calculate_multivariate_gaussian_distribution(mu, cov, vals)

    fig = plt.figure()
    cnf = fig.add_subplot(121)
    cond_dist = fig.add_subplot(122)

    cnf.contourf(X, Y, density, alpha=0.8)
    cnf.set_title("Multivariate Gaussian Distribution")
    cnf.set_xlabel("X")
    cnf.set_ylabel("Y")
    cnf.plot([-3, 3], [1.5, 1.5], color="red")

    # Calculate conditional Gaussian distribution
    y = 1.5
    mu_a, var_a = calculate_conditional_gaussian_distribution(mu, cov, y)
    cond_density = draw_gaussian_distribution.calculate_gaussian_distribution(
        X[0], mu_a, math.sqrt(var_a)
    )
    cond_dist.plot(X[0], cond_density)
    cond_dist_title = (
        "Conditional Gaussian Distribution: Y={0}, $\mu={1}, \sigma^2={2}$".format(
            y, mu_a, round(var_a, 2)
        )
    )
    cond_dist.set_title(cond_dist_title)
    cond_dist.set_xlabel("X")
    cond_dist.set_ylabel("Density")


if __name__ == "__main__":
    main()
    if len(sys.argv) == 2 and sys.argv[1] == "NotShow":
        print("It shows nothing")
    else:
        plt.show()
