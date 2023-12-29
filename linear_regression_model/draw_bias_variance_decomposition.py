import random
import sys

import matplotlib.pyplot as plt
import numpy as np


def calculate_gauss(x, mu, sigma=5):
    return np.exp(-((x - mu) ** 2) / 2 * sigma**2)


def calculate_gauss_mu(data_num, idx, range_start, range_end):
    interval = (range_end - range_start) / (data_num - 1)
    mu = range_start + interval * idx
    return mu


def update_weights(x, y, dim):
    A = np.zeros((dim + 1, dim + 1))
    T = np.zeros(dim + 1)
    for i in range(dim + 1):
        for j in range(dim + 1):
            multi_x = np.array(x) ** (i + j)
            A[i, j] = sum(multi_x)
        T[i] = np.dot(np.array(x) ** i, np.array(y))

    return np.dot(np.linalg.inv(A), T)


def predict(W, x):
    pred = np.zeros(x.shape)
    for i in range(len(W)):
        pred += np.dot(W[i], x**i)

    return pred


def calculate_design_matrix(noise_x, degree, range_start, range_end):
    design_mat = np.ones((len(noise_x), degree))
    for col in range(degree):
        if col == 0:
            continue
        else:
            mu = calculate_gauss_mu(degree-1, col-1, range_start, range_end)
            design_mat[:, col] = calculate_gauss(noise_x, mu)

    return design_mat


def predict_gauss(W, x, range_start, range_end):
    pred = np.zeros(x.shape)
    for i in range(len(W)):
        if i == 0:
            pred += np.dot(W[i], np.ones(x.shape))
        else:
            mu = calculate_gauss_mu(len(W)-1, i-1, range_start, range_end)
            pred += np.dot(W[i], calculate_gauss(x, mu))
            print(calculate_gauss(x, mu).shape)

    return pred

def main():
    # Create dataset
    random.seed(0)
    x = np.linspace(0, 2 * np.pi, 500)
    y = np.sin(x)
    range_start = -1
    range_end = 1
    N = 25
    L = 100
    pred_sum = np.zeros(x.shape)
    for i in range(L):
        nums = random.sample(range(x.shape[0]), k=N)
        # print(nums)
        noise_x = []
        noise_y = []
        for idx in nums:
            random_x = x[idx]
            random_y = y[idx] + np.random.normal(0, 0.2)
            noise_x.append(random_x)
            noise_y.append(random_y)

        degree = 3
        W = update_weights(np.array(noise_x), np.array(noise_y), degree)
        pred = predict(W, np.array(x))
        pred_sum += pred
        
        design_mat = calculate_design_matrix(np.array(noise_x), degree+1, range_start, range_end)
        print(design_mat)
        predict_gauss(W, np.array(x), range_start, range_end)
        input()

    pred_avg = pred_sum / L


    plt.plot(x, y, label="True")
    plt.plot(x, pred_avg, label="Prediction")
    plt.scatter(noise_x, noise_y, label="Noise points")
    plt.legend()
    title = "Polynomial curve fitting(degree={0},N={1})".format(str(degree), str(N))
    plt.title(title)


if __name__ == "__main__":
    main()
    if len(sys.argv) == 2 and sys.argv[1] == "NotShow":
        print("It shows nothing")
    else:
        plt.show()
