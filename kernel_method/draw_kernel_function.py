import random
import sys

import matplotlib.pyplot as plt
import numpy as np


def calculate_polynomial(x, degree):
    y = []
    for i in range(1, degree + 1):
        y.append(x**i)

    return y


def calculate_gauss(x, sigma, start, end, interval):
    gauss_y = []
    mu = start
    while mu <= end:
        gauss_y.append(np.exp(-((x - mu) ** 2) / 2 * sigma**2))
        mu += interval

    return gauss_y


def calculate_sigmoid(x, sigma, start, end, interval):
    sigmoid_y = []
    mu = start
    while mu <= end:
        sigmoid_y.append(1 / (1 + np.exp(-(x - mu) / sigma)))
        mu += interval

    return sigmoid_y


def main():
    # Create dataset
    random.seed(0)
    start = -1
    end = 1
    x = np.linspace(start, end, 500)

    fig = plt.figure(figsize=(16, 9))
    polynomial_graph = fig.add_subplot(231)
    gauss_graph = fig.add_subplot(232)
    sigmoid_graph = fig.add_subplot(233)
    polynomial_kernel_graph = fig.add_subplot(234)
    gauss_kernel_graph = fig.add_subplot(235)
    sigmoid_kernel_graph = fig.add_subplot(236)

    degree = 10
    poly_y = calculate_polynomial(x, degree)
    poly_kernel_x = -0.5
    poly_kernel_y = np.dot(
        np.array(calculate_polynomial(x, degree)).T,
        np.array(calculate_polynomial(np.array([poly_kernel_x]), degree)),
    )

    sigma = 5
    interval = 0.2
    gauss_y = calculate_gauss(x, sigma, start, end, interval)
    gauss_kernel_x = 0.0
    gauss_kernel_y = np.dot(
        np.array(calculate_gauss(x, sigma, start, end, interval)).T,
        np.array(
            calculate_gauss(np.array([gauss_kernel_x]), sigma, start, end, interval)
        ),
    )

    sigmoid_sigma = 0.2
    sigmoid_y = calculate_sigmoid(x, sigmoid_sigma, start, end, interval)
    sigmoid_kernel_x = 0.0
    sigmoid_kernel_y = np.dot(
        np.array(calculate_sigmoid(x, sigmoid_sigma, start, end, interval)).T,
        np.array(
            calculate_sigmoid(
                np.array([sigmoid_kernel_x]), sigmoid_sigma, start, end, interval
            )
        ),
    )

    for y in poly_y:
        polynomial_graph.plot(x, y)
        polynomial_graph.set_title("Polynomial")
    for y in gauss_y:
        gauss_graph.plot(x, y)
        gauss_graph.set_title("Gaussian")
    for y in sigmoid_y:
        sigmoid_graph.plot(x, y)
        sigmoid_graph.set_title("Sigmoid")

    # Draw kernel
    polynomial_kernel_graph.plot(x, poly_kernel_y)
    polynomial_kernel_graph.set_title(
        "Polynomial kernel($x'={0}$)".format(str(poly_kernel_x))
    )

    gauss_kernel_graph.plot(x, gauss_kernel_y)
    gauss_kernel_graph.set_title(
        "Gaussian kernel($x'={0}$)".format(str(gauss_kernel_x))
    )

    sigmoid_kernel_graph.plot(x, sigmoid_kernel_y)
    sigmoid_kernel_graph.set_title(
        "Sigmoid kernel($x'={0}$)".format(str(sigmoid_kernel_x))
    )


if __name__ == "__main__":
    main()
    if len(sys.argv) == 2 and sys.argv[1] == "NotShow":
        print("It shows nothing")
    else:
        plt.show()
