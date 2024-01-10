import math
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../")
from probability_distribution import draw_multivariate_gaussian_distribution


def create_data_space(rand_w, x_min, x_max):
    w_0 = rand_w[:, 0]
    w_1 = rand_w[:, 1]
    print(w_0)
    print(w_1)
    print(rand_w)
    for i in range(w_0.shape[0]):
        y_0 = w_0[i] * x_min + w_1[i]
        y_1 = w_0[i] * x_max + w_1[i]
        x_0 = x_min
        x_1 = x_max
        print(y_0)
        print(y_1)
        input()
        if y_0 > 1:
            y_0 = 1
            x_0 = (y_0 - w_1[i]) / w_0[i]
        elif y_0 < 0:
            y_0 = 0
            x_0 = (y_0 - w_1[i]) / w_0[i]
        if y_1 > 1:
            y_1 = 1
            x_1 = (y_1 - w_1[i]) / w_0[i]
        elif y_1 < 0:
            y_1 = 0
            x_1 = (y_1 - w_1[i]) / w_0[i]



def main():
    np.random.seed(314)

    fig = plt.figure()
    pri_post_dist = fig.add_subplot(121)
    data_space = fig.add_subplot(122)

    N = 100
    w_0 = np.linspace(-1, 1, N)
    w_1 = np.linspace(-1, 1, N)
    w_0, w_1 = np.meshgrid(w_0, w_1)

    mu = np.array([0, 0])
    cov = np.array([[0.2, 0.0], [0.0, 0.2]])

    vals = np.empty(w_0.shape + (2,))
    vals[:, :, 0] = w_0
    vals[:, :, 1] = w_1

    density = draw_multivariate_gaussian_distribution.calculate_multivariate_gaussian_distribution(mu, cov, vals)

    rand_w = np.random.multivariate_normal(mu, cov, size=6)
    print(rand_w)

    create_data_space(rand_w, data_space, -1, 1)

    pri_post_dist.contourf(w_0, w_1, density, alpha=0.8)
    pri_post_dist.set_title("Prior/Posterior distribution")
    pri_post_dist.set_xlabel("$w_0$")
    pri_post_dist.set_ylabel("$w_1$")



if __name__ == "__main__":
    main()
    if len(sys.argv) == 2 and sys.argv[1] == "NotShow":
        print("It shows nothing")
    else:
        plt.show()
