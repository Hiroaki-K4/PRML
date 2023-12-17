import matplotlib.pyplot as plt
import numpy as np
import random


def main():
    # Create dataset
    random.seed(0)
    x = np.linspace(0, 2*np.pi, 500)
    y = np.sin(x)
    nums = random.sample(range(x.shape[0]), k=10)
    print(type(nums))
    noise_x = []
    noise_y = []
    for idx in nums:
        print(idx)
        random_x = x[idx]
        random_y = y[idx] + np.random.normal(0, 0.3)
        print(random_x)
        print(random_y)
        noise_x.append(random_x)
        noise_y.append(random_y)

    M = 10
    A = np.empty((len(noise_x), M))
    print(A)
    print(A.shape)
    input()
    for i in range(len(noise_x)):
        for j in range(M):
            A[i, j] = noise_x[i] ** j

    print(A)
    w = np.dot(np.linalg.inv(A), np.array(noise_y))
    print("w: ", w)
    # TODO Predict y by using w

    plt.plot(x, y)
    plt.scatter(noise_x, noise_y)
    plt.show()


if __name__ == '__main__':
    main()
