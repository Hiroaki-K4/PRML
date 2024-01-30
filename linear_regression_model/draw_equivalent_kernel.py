import random
import sys

import matplotlib.pyplot as plt
import numpy as np

import draw_bias_variance_decomposition


def calculate_predicted_distributino_mean_variance(
    x, y, all_x, noise_precision, degree, range_start, range_end
):
    alpha = 2.0
    design_mat = draw_bias_variance_decomposition.calculate_design_matrix(
        x, degree + 1, range_start, range_end
    )
    s_n = np.linalg.pinv(
        alpha * np.identity(design_mat.shape[1])
        + noise_precision * np.dot(design_mat.T, design_mat)
    )
    all_input_design_mat = draw_bias_variance_decomposition.calculate_design_matrix(
        all_x, degree + 1, range_start, range_end
    )
    # Calculate mean of predicted distribution
    m_n = np.dot(
        np.dot(np.dot(noise_precision * s_n, design_mat.T), y), all_input_design_mat.T
    )

    # Calculate variacne of predicted distribution
    var_n = 1 / noise_precision + np.dot(
        np.dot(all_input_design_mat, s_n), all_input_design_mat.T
    )
    var_n = var_n[:, :1]

    return m_n, var_n


def draw_predicted_distribution(x, y, input_x, input_y, noise_precision, graph, N):
    # Plot input data
    graph.plot(x, y, c="blue")

    degree = 24
    range_start = -5
    range_end = 5
    m_n, var_n = calculate_predicted_distributino_mean_variance(
        np.array(input_x),
        np.array(input_y),
        x,
        noise_precision,
        degree,
        range_start,
        range_end,
    )

    graph.plot(x, m_n, c="red")
    max_y_list = []
    min_y_list = []
    for i in range(var_n.shape[0]):
        mean = float(m_n[i])
        var = float(var_n[i])
        max_y = mean + abs(var)
        min_y = mean - abs(var)
        max_y_list.append(max_y)
        min_y_list.append(min_y)

    # Draw predicted distribution
    graph.fill_between(x, min_y_list, max_y_list, color="orange")

    graph.scatter(input_x, input_y, c="blue")
    title = "$N={0}$".format(N)
    graph.set_title(title)
    graph.set_xlabel("$x$")
    graph.set_ylabel("$t$")


def create_random_input_data(x, y, input_x, input_y, N, noise_std):
    add_num = N - len(input_x)
    nums = random.sample(range(x.shape[0]), k=add_num)
    for idx in nums:
        random_x = x[idx]
        random_y = y[idx] + np.random.normal(0, noise_std)
        input_x.append(random_x)
        input_y.append(random_y)

    return input_x, input_y


def calculate_equivalent_kernel(
    x_0, x_1, noise_precision, degree, range_start, range_end
):
    alpha = 2.0
    design_mat_0 = draw_bias_variance_decomposition.calculate_design_matrix(
        np.flip(x_0), degree + 1, range_start, range_end
    )
    design_mat_1 = draw_bias_variance_decomposition.calculate_design_matrix(
        x_1, degree + 1, range_start, range_end
    )
    s_n = np.linalg.pinv(
        alpha * np.identity(design_mat_0.shape[1])
        + noise_precision * np.dot(design_mat_0.T, design_mat_0)
    )
    k = noise_precision * np.dot(np.dot(design_mat_0, s_n), design_mat_1.T)
    print(k)
    print(k.shape)
    # input()

    return k


def main():
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("Equivalent kernel")
    n_1_graph = fig.add_subplot(121)
    n_2_graph = fig.add_subplot(122)

    # Create dataset
    random.seed(0)
    x_0 = np.linspace(-1, 1, 200)
    tmp = np.flip(x_0)
    x_1 = np.linspace(-1, 1, 200)
    noise_std = 0.2
    noise_precision = 1 / (noise_std**2)
    degree = 24
    range_start = -5
    range_end = 5
    k = calculate_equivalent_kernel(
        x_0, x_1, noise_precision, degree, range_start, range_end
    )
    n_1_graph.contourf(x_0, x_1, k, alpha=0.8)


if __name__ == "__main__":
    main()
    if len(sys.argv) == 2 and sys.argv[1] == "NotShow":
        print("It shows nothing")
    else:
        plt.show()
