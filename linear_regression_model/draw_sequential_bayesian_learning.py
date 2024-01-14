import math
import random
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../")
from probability_distribution import (
    draw_gaussian_distribution,
    draw_multivariate_gaussian_distribution,
)


def normalize_line(x, y, w_0, w_1):
    norm_x = x
    norm_y = y
    if y > 1:
        norm_y = 1
        norm_x = (norm_y - w_1) / w_0
    elif y < -1:
        norm_y = -1
        norm_x = (norm_y - w_1) / w_0

    return norm_x, norm_y


def create_data_space(rand_w, x_min, x_max):
    w_0 = rand_w[:, 0]
    w_1 = rand_w[:, 1]
    data_space = np.empty((w_0.shape[0], 4))
    for i in range(w_0.shape[0]):
        y_0 = w_0[i] * x_min + w_1[i]
        y_1 = w_0[i] * x_max + w_1[i]
        x_0 = x_min
        x_1 = x_max
        x_0, y_0 = normalize_line(x_0, y_0, w_0[i], w_1[i])
        x_1, y_1 = normalize_line(x_1, y_1, w_0[i], w_1[i])
        data_space[i] = np.array([x_0, y_0, x_1, y_1])

    return data_space


def get_label(true_a_0, true_a_1, noise_std, x_min, x_max):
    x = random.uniform(x_min, x_max)
    y = true_a_0 * x + true_a_1
    noise = np.random.normal(0, noise_std)
    y += noise
    x, y = normalize_line(x, y, true_a_0, true_a_1)

    return x, y


def calculate_likelihood_function(x, y, w_s, noise_std):
    pred = np.empty((w_s.shape[0], w_s.shape[1]))
    for i in range(w_s.shape[0]):
        for j in range(w_s.shape[1]):
            pred[i][j] = w_s[i][j][0] * x + w_s[i][j][1]

    density = draw_gaussian_distribution.calculate_gaussian_distribution(
        y, pred, noise_std
    )

    return density


def draw_data_space(data_space_graph, rand_w, x_min, x_max):
    data_space = create_data_space(rand_w, x_min, x_max)
    for i in range(data_space.shape[0]):
        data_space_graph.plot(
            [data_space[i][0], data_space[i][2]],
            [data_space[i][1], data_space[i][3]],
            c="red",
        )

    data_space_graph.set_xlim(-1, 1)
    data_space_graph.set_ylim(-1, 1)
    data_space_graph.set_title("Data space")
    data_space_graph.set_xlabel("$x$")
    data_space_graph.set_ylabel("$y$")


def draw_posterior_distribution(post_graph, w_0, w_1, post_density):
    post_graph.contourf(w_0, w_1, post_density, alpha=0.8)
    post_graph.set_xlim(-1, 1)
    post_graph.set_ylim(-1, 1)
    post_graph.set_xlabel("$w_0$")
    post_graph.set_ylabel("$w_1$")


def draw_likelihood(likelihood_graph, w_0, w_1, w_density, true_a_0, true_a_1):
    likelihood_graph.set_title("Likelihood")
    likelihood_graph.contourf(w_0, w_1, w_density, alpha=0.8)
    likelihood_graph.scatter([true_a_0], [true_a_1], c="white")
    likelihood_graph.set_xlim(-1, 1)
    likelihood_graph.set_ylim(-1, 1)
    likelihood_graph.set_xlabel("$w_0$")
    likelihood_graph.set_ylabel("$w_1$")


def main():
    np.random.seed(314)
    random.seed(314)

    fig, (
        (ax1, pri_post_dist, data_space_graph),
        (likelihood_graph, ax4, ax5),
    ) = plt.subplots(2, 3)
    fig.delaxes(ax1)

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

    w_s = np.empty(w_0.shape + (2,))
    w_s[:, :, 0] = w_0
    w_s[:, :, 1] = w_1

    # Prior distribution
    density = draw_multivariate_gaussian_distribution.calculate_multivariate_gaussian_distribution(
        mu, cov, w_s
    )
    pri_post_dist.contourf(w_0, w_1, density, alpha=0.8)
    pri_post_dist.set_title("Prior/Posterior")
    pri_post_dist.set_xlabel("$w_0$")
    pri_post_dist.set_ylabel("$w_1$")

    # First data space
    rand_w = np.random.multivariate_normal(mu, cov, size=6)
    draw_data_space(data_space_graph, rand_w, x_min, x_max)

    # Add loop

    x, y = get_label(true_a_0, true_a_1, noise_std, x_min, x_max)
    w_density = calculate_likelihood_function(x, y, w_s, noise_std)
    draw_likelihood(likelihood_graph, w_0, w_1, w_density, true_a_0, true_a_1)

    post_density = density * w_density
    draw_posterior_distribution(ax4, w_0, w_1, post_density)


if __name__ == "__main__":
    main()
    if len(sys.argv) == 2 and sys.argv[1] == "NotShow":
        print("It shows nothing")
    else:
        plt.show()
