import random
import sys

import matplotlib.pyplot as plt
import numpy as np


def create_dataset(N):
    random.seed(314)
    x = np.linspace(0, 1, 100)
    y = x + 0.3 * np.sin(2 * np.pi * x)
    nums = random.sample(range(x.shape[0]), k=N)
    noise_x = []
    noise_y = []
    for idx in nums:
        random_x = x[idx]
        random_y = y[idx] + np.random.normal(0, 0.07)
        noise_x.append(random_x)
        noise_y.append(random_y)

    return x, y, noise_x, noise_y


def main():
    N = 100
    x, y, noise_x, noise_y = create_dataset(N)

    degree = 3

    plt.plot(y, x, label="True")
    plt.scatter(noise_y, noise_x, label="Noise points", color="green")
    plt.legend()
    title = "Polynomial curve fitting(degree={0},N={1})".format(str(degree), str(N))
    plt.title(title)


if __name__ == "__main__":
    main()
    if len(sys.argv) == 2 and sys.argv[1] == "NotShow":
        print("It shows nothing")
    else:
        plt.show()
