import sys

import matplotlib.pyplot as plt
import numpy as np


def create_dataset(N):
    x = np.linspace(-2, 2, N)
    sigmoid = 1 / (1 + np.exp(-x))
    tanh = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    return x, sigmoid, tanh


def main():
    x, sigmoid, tanh = create_dataset(100)
    plt.plot(x, sigmoid, label="sigmoid")
    plt.plot(x, tanh, label="tanh")
    plt.legend()
    title = "Comparison of tanh and sigmoid function"
    plt.title(title)


if __name__ == "__main__":
    main()
    if len(sys.argv) == 2 and sys.argv[1] == "NotShow":
        print("It shows nothing")
    else:
        plt.show()
