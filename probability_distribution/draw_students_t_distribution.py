import math
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma

import draw_gaussian_distribution


def calculate_students_t_distribution(vals, mu, prec, nu):
    density = gamma((nu+1)/2) / gamma(nu/2) * (prec/(np.pi*nu))**0.5 * (1+prec*(vals-mu)**2)**((-nu-1)/2)

    return density


def main():
    mu = 0
    var = 1.0
    precision = 1 / np.sqrt(var)
    x_vals = np.linspace(mu - var * 4.0, mu + var * 4.0, num=250)
    high_nu_density = calculate_students_t_distribution(x_vals, mu, precision, 5.0)
    middle_nu_density = calculate_students_t_distribution(x_vals, mu, precision, 1.0)
    low_nu_density = calculate_students_t_distribution(x_vals, mu, precision, 0.1)

    plt.plot(x_vals, high_nu_density, label=r"$\nu=5.0$")
    plt.plot(x_vals, middle_nu_density, label=r"$\nu=1.0$")
    plt.plot(x_vals, low_nu_density, label=r"$\nu=0.1$")
    plt.legend()

    plt.figure()
    plt.plot(range(10), 'ro-')


if __name__ == "__main__":
    main()
    if len(sys.argv) == 2 and sys.argv[1] == "NotShow":
        print("It shows nothing")
    else:
        plt.show()
