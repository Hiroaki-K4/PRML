import sys

import matplotlib.pyplot as plt
import numpy as np


def create_dataset(N):
    x = np.linspace(-2.5, 2.5, N)
    sinh = (np.exp(x) - np.exp(-x)) / 2
    cosh = (np.exp(x) + np.exp(-x)) / 2
    tanh = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    return x, sinh, cosh, tanh


def main():
    x, sinh, cosh, tanh = create_dataset(500)
    plt.plot(x, sinh, label="sinh")
    plt.plot(x, cosh, label="cosh")
    plt.plot(x, tanh, label="tanh")
    plt.legend()
    title = "Hyperbolic functions"
    plt.title(title)


if __name__ == "__main__":
    main()
    if len(sys.argv) == 2 and sys.argv[1] == "NotShow":
        print("It shows nothing")
    else:
        plt.show()
