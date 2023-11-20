import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad


def integrand(theta, m):
    return np.exp(m * np.cos(theta))


def calculate_von_mises_distribution(theta_vals, m, theta_0):
    norm_fac = 1 / 2 * np.pi * quad(integrand, 0, 2 * np.pi, args=(m))[0]
    density = 1 / (2 * np.pi * norm_fac) * np.exp(m * np.cos(theta_vals - theta_0))

    return density


def main():
    theta_vals = np.linspace(0, 2 * np.pi, num=250)
    m = 5
    theta_0 = np.pi / 4
    density = calculate_von_mises_distribution(theta_vals, m, theta_0)
    print(density)
    plt.plot(theta_vals, density, label=r"$m=5,\theta_0=\pi/4$")

    m = 1
    theta_0 = 3 * np.pi / 4
    density = calculate_von_mises_distribution(theta_vals, m, theta_0)
    print(density)
    plt.plot(theta_vals, density, label=r"$m=1,\theta_0=3\pi/4$")
    plt.legend()
    plt.title("von Mises distribution")


if __name__ == "__main__":
    main()
    if len(sys.argv) == 2 and sys.argv[1] == "NotShow":
        print("It shows nothing")
    else:
        plt.show()
