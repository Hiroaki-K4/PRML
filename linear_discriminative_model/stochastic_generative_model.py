import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../probability_distribution")
from draw_multivariate_gaussian_distribution import \
    calculate_multivariate_gaussian_distribution


def main():
    np.random.seed(314)
    N = 100
    X = np.linspace(-3, 3, N)
    Y = np.linspace(-3, 3, N)
    X, Y = np.meshgrid(X, Y)
    vals = np.empty(X.shape + (2,))
    vals[:, :, 0] = X
    vals[:, :, 1] = Y
    print(vals[:, :, 0])
    print(np.reshape(vals[:, :, 0], 10000))
    print(np.reshape(vals[:, :, 0], 10000).shape)
    input()

    # Create dataset
    mean_green = np.array([-1.0, 1.0])
    cov_green = np.array([[0.5, 0.0], [0.0, 0.1]])
    cluster_green = np.random.multivariate_normal(mean_green, cov_green, 5000)
    green_density = calculate_multivariate_gaussian_distribution(
        mean_green, cov_green, vals
    )
    print(green_density)
    input()

    mean_red = np.array([-1.0, -1.0])
    cov_red = np.array([[0.5, 0.0], [0.0, 0.1]])
    cluster_red = np.random.multivariate_normal(mean_red, cov_red, 5000)
    red_density = calculate_multivariate_gaussian_distribution(mean_red, cov_red, vals)

    mean_blue = np.array([1.5, 0.0])
    cov_blue = np.array([[0.1, 0.0], [0.0, 0.5]])
    cluster_blue = np.random.multivariate_normal(mean_blue, cov_blue, 5000)
    red_density = calculate_multivariate_gaussian_distribution(
        mean_blue, cov_blue, vals
    )

    # TODO Add calculation that determin RGB

    fig = plt.figure(figsize=(16, 9))
    linear_graph_0 = fig.add_subplot(121)
    linear_graph_0.scatter(cluster_green[:, 0], cluster_green[:, 1], c="green")
    linear_graph_0.scatter(cluster_red[:, 0], cluster_red[:, 1], c="red")
    linear_graph_0.scatter(cluster_blue[:, 0], cluster_blue[:, 1], c="blue")
    linear_graph_0.set_title("Not Fisher's linear discriminant")


if __name__ == "__main__":
    main()
    if len(sys.argv) == 2 and sys.argv[1] == "NotShow":
        print("It shows nothing")
    else:
        plt.show()
