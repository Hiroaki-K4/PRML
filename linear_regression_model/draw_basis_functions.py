import random
import sys

import matplotlib.pyplot as plt
import numpy as np


def main():
    # Create dataset
    random.seed(0)
    x = np.linspace(-1, 1, 500)
    print(x)
    input()

    # TODO Draw 3 basis functions
    # Draw polynomial
    # Draw Gauss basis function
    # Draw sigmoid function


    # plt.plot(x, y, label="True")
    # plt.plot(x, pred, label="Prediction")
    # plt.scatter(noise_x, noise_y, label="Noise points")
    # plt.legend()
    # title = "Polynomial curve fitting(degree={0},N={1})".format(str(degree), str(N))
    # plt.title(title)


if __name__ == "__main__":
    main()
    if len(sys.argv) == 2 and sys.argv[1] == "NotShow":
        print("It shows nothing")
    else:
        plt.show()
