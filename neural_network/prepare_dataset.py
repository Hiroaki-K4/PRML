import random
import sys

import matplotlib.pyplot as plt
import numpy as np


def create_dataset(N):
    x = np.linspace(0, 1, N)
    y = x + 0.3 * np.sin(2 * np.pi * x)
    nums = random.sample(range(x.shape[0]), k=N)
    noise_x = []
    noise_y = []
    for idx in nums:
        random_x = x[idx]
        random_y = y[idx] + np.random.normal(0, 0.07)
        noise_x.append(random_x)
        noise_y.append(random_y)

    return y, x, noise_y, noise_x


def main():
    random.seed(314)
    np.random.seed(314)

    N = 100
    x, y, input_x, input_y = create_dataset(N)

    print(input_x)
    print(input_y)
    input()
    plt.plot(x, y, label="True")
    plt.scatter(input_x, input_y, label="Input points", color="green")
    plt.legend()
    title = "Mixed density network(N={0})".format(str(N))
    plt.title(title)


if __name__ == "__main__":
    main()
    if len(sys.argv) == 2 and sys.argv[1] == "NotShow":
        print("It shows nothing")
    else:
        plt.show()
