import numpy as np
import matplotlib.pyplot as plt
from   mpl_toolkits.mplot3d import Axes3D
from   scipy.stats import multivariate_normal


def calculate_multi_dimentional_gaussian_distribution(mu, cov, vals):
    print("ok")
    # TODO Add gaussin distribution 


def main():
    N = 100
    X = np.linspace(-3, 3, N)
    Y = np.linspace(-3, 3, N)
    X, Y = np.meshgrid(X, Y)

    mu = np.array([0, 0])
    cov = np.array([[ 1. , 0.5], [0.5,  1.5]])

    vals = np.empty(X.shape + (2,))
    vals[:, :, 0] = X
    vals[:, :, 1] = Y

    F = multivariate_normal(mu, cov)
    Z = F.pdf(vals)
    prob = calculate_multi_dimentional_gaussian_distribution(mu, cov, vals)


    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis")

    plt.show()


if __name__ == "__main__":
    main()
