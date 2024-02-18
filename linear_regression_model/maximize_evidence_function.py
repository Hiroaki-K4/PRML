import random

import numpy as np

from evaluate_evidence_function import calculate_design_matrix_using_polynomial


def predict_hyperparameters(x, y, alpha, beta, degree):
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

        err = 0
        for idx in range(design_mat.shape[0]):
            err += (y[idx] - np.dot(m_N.T, design_mat[idx, :])) ** 2

        new_alpha = gamma / np.dot(m_N.T, m_N)
        new_beta = 1 / (design_mat.shape[0] - gamma) * err
        new_beta = 1 / new_beta

        if abs(alpha - new_alpha) < 1e-3 and abs(beta - new_beta) < 1e-3:
            alpha = new_alpha
            beta = new_beta
            break

        alpha = new_alpha
        beta = new_beta

    return alpha, beta


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

    # Initial value
    pred_alpha = 0.02
    pred_beta = 15
    degree = 9
    pred_alpha, pred_beta = predict_hyperparameters(
        noise_x, noise_y, pred_alpha, pred_beta, degree
    )
    print("----- Predicted hyperparameters -----")
    print("alpha: ", pred_alpha)
    print("beta: ", pred_beta)


if __name__ == "__main__":
    main()
