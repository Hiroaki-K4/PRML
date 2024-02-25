import sys

import matplotlib.pyplot as plt
import numpy as np


def calculate_decision_boundary_by_fishers_linear_discriminant(cluster_0, cluster_1):
    m_0 = sum(cluster_0) / len(cluster_0)
    m_1 = sum(cluster_1) / len(cluster_1)
    Sw = np.zeros((2, 2))
    for i in range(cluster_0.shape[0]):
        diff_0 = np.reshape(cluster_0[i] - m_0, (2, 1))
        diff_1 = np.reshape(cluster_1[i] - m_1, (2, 1))
        Sw += np.dot(diff_0, diff_0.T) + np.dot(diff_1, diff_1.T)

    w = np.dot(np.linalg.inv(Sw), m_1 - m_0)
    mean = (m_0 + m_1) / 2

    return w, mean, m_0, m_1


def get_line_by_decision_boundary_slope(vec, mean):
    a = -vec[0] / vec[1]
    b = -a * mean[0] + mean[1]
    x = np.linspace(-2, 6, 800)
    y = a * x + b

    return x, y


def main():
    np.random.seed(314)

    # Create dataset
    mean = np.array([1.0, 3.0])
    cov = np.array([[4.0, 0.5], [0.5, 0.15]])
    cluster_0 = np.random.multivariate_normal(mean, cov, 50)
    mean = np.array([4.0, 2.0])
    cluster_1 = np.random.multivariate_normal(mean, cov, 50)

    (
        w,
        clust_mean,
        m_0,
        m_1,
    ) = calculate_decision_boundary_by_fishers_linear_discriminant(cluster_0, cluster_1)

    # Draw decision boundary
    m_vec = m_1 - m_0
    x, y = get_line_by_decision_boundary_slope(m_vec, clust_mean)

    # Draw decision boundary by Fisher's linear discriminant
    x_fisher, y_fisher = get_line_by_decision_boundary_slope(w, clust_mean)

    fig = plt.figure(figsize=(16, 9))
    linear_graph_0 = fig.add_subplot(121)
    linear_graph_0.scatter(cluster_0[:, 0], cluster_0[:, 1], c="blue")
    linear_graph_0.scatter(cluster_1[:, 0], cluster_1[:, 1], c="red")
    linear_graph_0.scatter(x, y, s=3.0, c="green")
    linear_graph_0.set_title("Not Fisher's linear discriminant")

    linear_graph_1 = fig.add_subplot(122)
    linear_graph_1.scatter(cluster_0[:, 0], cluster_0[:, 1], c="blue")
    linear_graph_1.scatter(cluster_1[:, 0], cluster_1[:, 1], c="red")
    linear_graph_1.scatter(x_fisher, y_fisher, s=3.0, c="green")
    linear_graph_1.set_title("Fisher's linear discriminant")


if __name__ == "__main__":
    main()
    if len(sys.argv) == 2 and sys.argv[1] == "NotShow":
        print("It shows nothing")
    else:
        plt.show()
