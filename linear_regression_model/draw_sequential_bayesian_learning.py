import math
import random
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../")
from probability_distribution import draw_multivariate_gaussian_distribution


def create_data_space(rand_w, x_min, x_max):
    w_0 = rand_w[:, 0]
    w_1 = rand_w[:, 1]
    data_space = np.empty((w_0.shape[0], 4))
    for i in range(w_0.shape[0]):
        y_0 = w_0[i] * x_min + w_1[i]
        y_1 = w_0[i] * x_max + w_1[i]
        x_0 = x_min
        x_1 = x_max
        if y_0 > 1:
            y_0 = 1
            x_0 = (y_0 - w_1[i]) / w_0[i]
        elif y_0 < -1:
            y_0 = -1
            x_0 = (y_0 - w_1[i]) / w_0[i]
        if y_1 > 1:
            y_1 = 1
            x_1 = (y_1 - w_1[i]) / w_0[i]
        elif y_1 < -1:
            y_1 = -1
            x_1 = (y_1 - w_1[i]) / w_0[i]

        data_space[i] = np.array([x_0, y_0, x_1, y_1])

    return data_space


def get_label(true_a_0, true_a_1, noise_std, x_min, x_max):
    x = random.uniform(x_min, x_max)
    y = true_a_0 * x + true_a_1
    noise = noise = np.random.normal(0, noise_std)

    return x, y + noise


def main():
    np.random.seed(314)

    fig, (
        (ax1, pri_post_dist, data_space_graph),
        (likelihood_graph, ax4, ax5),
    ) = plt.subplots(2, 3)
    data_space_graph.set_xlim(-1, 1)
    data_space_graph.set_ylim(-1, 1)
    likelihood_graph.set_xlim(-1, 1)
    likelihood_graph.set_ylim(-1, 1)

    true_a_0 = -0.3
    true_a_1 = 0.5
    noise_std = 0.2
    x_min = -1
    x_max = 1

    N = 100
    w_0 = np.linspace(-1, 1, N)
    w_1 = np.linspace(-1, 1, N)
    w_0, w_1 = np.meshgrid(w_0, w_1)

    mu = np.array([0, 0])
    cov = np.array([[0.2, 0.0], [0.0, 0.2]])

    vals = np.empty(w_0.shape + (2,))
    vals[:, :, 0] = w_0
    vals[:, :, 1] = w_1

    density = draw_multivariate_gaussian_distribution.calculate_multivariate_gaussian_distribution(
        mu, cov, vals
    )

    rand_w = np.random.multivariate_normal(mu, cov, size=6)

    data_space = create_data_space(rand_w, x_min, x_max)
    for i in range(data_space.shape[0]):
        data_space_graph.plot(
            [data_space[i][0], data_space[i][2]],
            [data_space[i][1], data_space[i][3]],
            c="red",
        )

    x, y = get_label(true_a_0, true_a_1, noise_std, x_min, x_max)
    likelihood_graph.scatter([x], [y])

    # TODO calculate likelihood function

    pri_post_dist.contourf(w_0, w_1, density, alpha=0.8)
    pri_post_dist.set_title("Prior/Posterior distribution")
    pri_post_dist.set_xlabel("$w_0$")
    pri_post_dist.set_ylabel("$w_1$")
    data_space_graph.set_xlabel("$x$")
    data_space_graph.set_ylabel("$y$")
    likelihood_graph.set_xlabel("$w_0$")
    likelihood_graph.set_ylabel("$w_1$")


if __name__ == "__main__":
    main()
    if len(sys.argv) == 2 and sys.argv[1] == "NotShow":
        print("It shows nothing")
    else:
        plt.show()
