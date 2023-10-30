import sys

import matplotlib.pyplot as plt
import numpy as np


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


def calculate_conditional_gaussian_distribution(target_idx, mu, cov, y):
    pre_mat = np.linalg.inv(cov)
    print("target: ", pre_mat[target_idx, target_idx]**(-1))
    var = np.linalg.inv(pre_mat[target_idx, target_idx])
    print("cov: ", cov)
    print("pre_mat: ", pre_mat)
    print("var: ", var)

    return 0


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
    fig.suptitle(t="Multivariate gaussian distribution", fontsize=20)
    cnf = plt.contourf(X, Y, density, alpha=0.8)
    plt.colorbar(cnf, label="density")

    # Calculate conditional Gaussian distribution
    y = 1.5
    a = calculate_conditional_gaussian_distribution(0, mu, cov, y)


if __name__ == "__main__":
    main()
    if len(sys.argv) == 2 and sys.argv[1] == "NotShow":
        print("It shows nothing")
    else:
        plt.show()
