import math
import random

import numpy as np


def calculate_design_matrix_using_polynomial(x, degree):
    design_mat = np.ones((len(x), degree))
    for col in range(degree):
        design_mat[:, col] = np.array(x) ** col

    return design_mat


def calculate_evidence_function(x, y, degree, alpha, beta):
    y_arr = np.array(y)
    design_mat = calculate_design_matrix_using_polynomial(x, degree)
    M = degree - 1
    N = y_arr.shape[0]
    A = alpha * np.identity(degree) + beta * np.dot(design_mat.T, design_mat)
    m_N = beta * np.dot(np.dot(np.linalg.inv(A), design_mat.T), y_arr)
    E_m_N = beta / 2 * np.linalg.norm(
        y_arr - np.dot(design_mat, m_N)
    ) ** 2 + alpha / 2 * np.dot(m_N.T, m_N)
    evidence_func = (
        M / 2 * math.log(alpha)
        + N / 2 * math.log(beta)
        - E_m_N
        - 1 / 2 * math.log(np.linalg.det(A))
        - N / 2 * math.log(2 * math.pi)
    )

    return evidence_func


def predict_alpha(x, y, alpha, beta, degree):
    design_mat = calculate_design_matrix_using_polynomial(x, degree)
    y_arr = np.array(y)
    while True:
        A = alpha * np.identity(degree) + beta * np.dot(design_mat.T, design_mat)
        m_N = beta * np.dot(np.dot(np.linalg.inv(A), design_mat.T), y_arr)

        eig = beta * np.dot(design_mat.T, design_mat)
        eigen_values, eigen_vectors = np.linalg.eig(eig)
        gamma = 0
        for i in range(eigen_values.shape[0]):
            gamma += eigen_values[i] / (alpha + eigen_values[i])

        new_alpha = gamma / np.dot(m_N.T, m_N)
        if abs(alpha - new_alpha) < 1e-4:
            alpha = new_alpha
            break

        alpha = new_alpha

    return alpha


def main():
    # Create dataset
    random.seed(314)
    np.random.seed(314)
    x = np.linspace(0, 2 * np.pi, 500)
    y = np.sin(x)
    N = 30
    nums = random.sample(range(x.shape[0]), k=N)
    noise_std = 0.2
    noise_x = []
    noise_y = []
    for idx in nums:
        random_x = x[idx]
        random_y = y[idx] + np.random.normal(0, noise_std)
        noise_x.append(random_x)
        noise_y.append(random_y)

    true_alpha = 0.005
    true_beta = (1 / 0.2) ** 2

    pred_alpha = 0.005
    pred_beta = 23
    degree = 9
    pred_alpha = predict_alpha(noise_x, noise_y, pred_alpha, pred_beta, degree)
    print(pred_alpha)


if __name__ == "__main__":
    main()
