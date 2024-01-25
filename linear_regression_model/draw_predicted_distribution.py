import random
import sys

import matplotlib.pyplot as plt
import numpy as np


def calculate_gauss(x, mu, sigma=1.5):
    return np.exp(-((x - mu) ** 2) / 2 * sigma**2)


def calculate_gauss_mu(data_num, idx, range_start, range_end):
    interval = (range_end - range_start) / (data_num - 1)
    mu = range_start + interval * idx
    return mu


def update_weights(design_mat, y, lam):
    W = np.dot(
        np.dot(
            np.linalg.inv(
                np.dot(lam, np.identity(int(y.shape[0])))
                + np.dot(design_mat.T, design_mat)
            ),
            design_mat.T,
        ),
        y,
    )
    return W


def calculate_design_matrix(noise_x, degree, range_start, range_end):
    design_mat = np.ones((len(noise_x), degree))
    for col in range(degree):
        if col == 0:
            continue
        else:
            mu = calculate_gauss_mu(degree - 1, col - 1, range_start, range_end)
            design_mat[:, col] = calculate_gauss(noise_x, mu)

    return design_mat


def predict(W, x, range_start, range_end):
    pred = np.zeros(x.shape)
    for i in range(len(W)):
        if i == 0:
            pred += np.dot(W[i], np.ones(x.shape))
        else:
            mu = calculate_gauss_mu(len(W) - 1, i - 1, range_start, range_end)
            pred += np.dot(W[i], calculate_gauss(x, mu))

    return pred


def draw_predicted_distribution(x, y, input_x, input_y, graph):
    graph.plot(x, y)
    graph.scatter(input_x, input_y)


def create_random_input_data(x, y, input_x, input_y, N):
    add_num = N - len(input_x)
    nums = random.sample(range(x.shape[0]), k=add_num)
    for idx in nums:
        random_x = x[idx]
        random_y = y[idx] + np.random.normal(0, 0.01)
        input_x.append(random_x)
        input_y.append(random_y)

    return input_x, input_y


def main():
    fig = plt.figure(figsize=(16, 9))
    n_1_graph = fig.add_subplot(221)
    n_2_graph = fig.add_subplot(222)
    n_4_graph = fig.add_subplot(223)
    n_25_graph = fig.add_subplot(224)

    # Create dataset
    random.seed(0)
    x = np.linspace(0, 2 * np.pi, 500)
    y = np.sin(x)
    range_start = -5
    range_end = 5
    N = 25
    L = 100
    input_x = []
    input_y = []
    N = 1
    input_x, input_y = create_random_input_data(x, y, input_x, input_y, 1)
    draw_predicted_distribution(x, y, input_x, input_y, n_1_graph)
    input_x, input_y = create_random_input_data(x, y, input_x, input_y, 2)
    draw_predicted_distribution(x, y, input_x, input_y, n_2_graph)
    input_x, input_y = create_random_input_data(x, y, input_x, input_y, 4)
    draw_predicted_distribution(x, y, input_x, input_y, n_4_graph)
    input_x, input_y = create_random_input_data(x, y, input_x, input_y, 25)
    draw_predicted_distribution(x, y, input_x, input_y, n_25_graph)


if __name__ == "__main__":
    main()
    if len(sys.argv) == 2 and sys.argv[1] == "NotShow":
        print("It shows nothing")
    else:
        plt.show()
