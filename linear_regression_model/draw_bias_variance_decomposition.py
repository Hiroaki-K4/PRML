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


def main():
    fig = plt.figure(figsize=(16, 9))
    lam_1_graph = fig.add_subplot(221)
    lam_1_avg_graph = fig.add_subplot(222)
    lam_2_graph = fig.add_subplot(223)
    lam_2_avg_graph = fig.add_subplot(224)

    # Create dataset
    random.seed(0)
    x = np.linspace(0, 2 * np.pi, 500)
    y = np.sin(x)
    range_start = -5
    range_end = 5
    N = 25
    L = 100
    pred_sum_1 = np.zeros(x.shape)
    pred_sum_2 = np.zeros(x.shape)
    lambda_1 = 2.6
    lambda_2 = -0.31
    for i in range(L):
        nums = random.sample(range(x.shape[0]), k=N)
        noise_x = []
        noise_y = []
        for idx in nums:
            random_x = x[idx]
            random_y = y[idx] + np.random.normal(0, 0.01)
            noise_x.append(random_x)
            noise_y.append(random_y)

        degree = 24
        design_mat = calculate_design_matrix(
            np.array(noise_x), degree + 1, range_start, range_end
        )
        W_1 = update_weights(design_mat, np.array(noise_y), lambda_1)
        W_2 = update_weights(design_mat, np.array(noise_y), lambda_2)
        pred_1 = predict(W_1, np.array(x), range_start, range_end)
        pred_2 = predict(W_2, np.array(x), range_start, range_end)
        pred_sum_1 += pred_1
        pred_sum_2 += pred_2
        lam_1_graph.plot(x, pred_1)
        lam_2_graph.plot(x, pred_2)

    title = "Predictions($\lambda$={0},L={1})".format(str(lambda_1), str(L))
    lam_1_graph.set_title(title)
    title = "Predictions($\lambda$={0},L={1})".format(str(lambda_2), str(L))
    lam_2_graph.set_title(title)

    pred_avg_1 = pred_sum_1 / L
    pred_avg_2 = pred_sum_2 / L
    lam_1_avg_graph.plot(x, pred_avg_1, label="Prediction")
    lam_1_avg_graph.plot(x, y, label="True", c="red", linestyle="--")
    lam_1_avg_graph.set_title("Average of prediction")
    lam_2_avg_graph.plot(x, pred_avg_2, label="Prediction")
    lam_2_avg_graph.plot(x, y, label="True", c="red", linestyle="--")
    lam_2_avg_graph.set_title("Average of prediction")
    lam_1_avg_graph.legend()
    lam_2_avg_graph.legend()


if __name__ == "__main__":
    main()
    if len(sys.argv) == 2 and sys.argv[1] == "NotShow":
        print("It shows nothing")
    else:
        plt.show()
