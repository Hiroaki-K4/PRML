import sys

import matplotlib.pyplot as plt
import numpy as np

import draw_bias_variance_decomposition


def find_nearest_value_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()

    return idx


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
    kernel = noise_precision * np.dot(np.dot(design_mat_0, s_n), design_mat_1.T)

    return kernel


def draw_kernel_intercept(kernel, idx, x, graph, color, intercept):
    interc = kernel[idx, :]
    graph.plot(x, interc, color=color)
    title = "$x={0}$".format(intercept)
    graph.set_title(title)


def main():
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("Equivalent kernel")
    graph_0 = fig.add_subplot(221)
    graph_1 = fig.add_subplot(222)
    graph_2 = fig.add_subplot(223)
    graph_3 = fig.add_subplot(224)

    # Create dataset
    x_0 = np.linspace(-1, 1, 200)
    x_1 = np.linspace(-1, 1, 200)
    noise_std = 0.2
    noise_precision = 1 / (noise_std**2)
    degree = 24
    range_start = -5
    range_end = 5

    # Calculate equivalent kernel
    kernel = calculate_equivalent_kernel(
        x_0, x_1, noise_precision, degree, range_start, range_end
    )
    graph_0.contourf(x_1, np.flip(x_1), kernel)

    # Draw intercepts
    x_0_0 = 0.85
    idx = find_nearest_value_idx(np.flip(x_1), x_0_0)
    draw_kernel_intercept(kernel, idx, x_1, graph_1, "red", x_0_0)
    graph_0.plot([-1, 1], [x_0_0, x_0_0], color="red")

    x_0_1 = 0.0
    idx = find_nearest_value_idx(np.flip(x_1), x_0_1)
    draw_kernel_intercept(kernel, idx, x_1, graph_2, "green", x_0_1)
    graph_0.plot([-1, 1], [x_0_1, x_0_1], color="green")

    x_0_2 = -0.75
    idx = find_nearest_value_idx(np.flip(x_1), x_0_2)
    draw_kernel_intercept(kernel, idx, x_1, graph_3, "blue", x_0_2)
    graph_0.plot([-1, 1], [x_0_2, x_0_2], color="blue")

    graph_0.set_xlabel("$x'$")
    graph_0.set_ylabel("$x$")
    graph_1.set_xlabel("$x'$")
    graph_1.set_ylabel("$x$")
    graph_2.set_xlabel("$x'$")
    graph_2.set_ylabel("$x$")
    graph_3.set_xlabel("$x'$")
    graph_3.set_ylabel("$x$")


if __name__ == "__main__":
    main()
    if len(sys.argv) == 2 and sys.argv[1] == "NotShow":
        print("It shows nothing")
    else:
        plt.show()
