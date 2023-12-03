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


def main():
    N = 100
    X = np.linspace(-8, 8, N)
    Y = np.linspace(-8, 8, N)
    X, Y = np.meshgrid(X, Y)

    mu_1 = np.array([0, 0])
    cov_1 = np.array([[1.0, 0.5], [0.5, 1.5]])
    mu_2 = np.array([2.5, 2.5])
    cov_2 = np.array([[1.0, 0.5], [0.5, 1.5]])
    mu_3 = np.array([-2.5, -2.5])
    cov_3 = np.array([[1.0, 0.5], [0.5, 1.5]])

    vals = np.empty(X.shape + (2,))
    vals[:, :, 0] = X
    vals[:, :, 1] = Y

    mix_coef_1 = 0.5
    mix_coef_2 = 0.2
    mix_coef_3 = 0.3
    if mix_coef_1 + mix_coef_2 + mix_coef_3 != 1.0:
        print("Sum of mixing coefficient is not 1.")
        return
    if (
        (mix_coef_1 < 0 or mix_coef_1 > 1)
        or (mix_coef_2 < 0 or mix_coef_2 > 1)
        or (mix_coef_3 < 0 or mix_coef_3 > 1)
    ):
        print("Mixing coefficient is wrong.")
        return

    density_1 = calculate_multivariate_gaussian_distribution(mu_1, cov_1, vals)
    density_2 = calculate_multivariate_gaussian_distribution(mu_2, cov_2, vals)
    density_3 = calculate_multivariate_gaussian_distribution(mu_3, cov_3, vals)
    mix_density = (
        mix_coef_1 * density_1 + mix_coef_2 * density_2 + mix_coef_3 * density_3
    )

    fig = plt.figure()
    fig.suptitle(t="Mixture of Gaussians", fontsize=20)
    ax = fig.add_subplot(projection="3d")
    ax.set_xlabel(xlabel="Dimention 1")
    ax.set_ylabel(ylabel="Dimention 2")
    ax.set_zlabel(zlabel="Density")
    param_text_1 = (
        "$\\mu_1=({0},{1}), \\mu_2=({2},{3}), \\mu_3=({4},{5})$, cov={6}".format(
            mu_1[0], mu_1[1], mu_2[0], mu_2[1], mu_3[0], mu_3[1], cov_1
        )
    )
    ax.set_title(label=param_text_1, loc="left")
    ax.plot_surface(X, Y, mix_density, cmap="viridis")


if __name__ == "__main__":
    main()
    if len(sys.argv) == 2 and sys.argv[1] == "NotShow":
        print("It shows nothing")
    else:
        plt.show()
