import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma

import draw_gaussian_distribution


def calculate_students_t_distribution(vals, mu, prec, nu):
    density = (
        gamma((nu + 1) / 2)
        / gamma(nu / 2)
        * (prec / (np.pi * nu)) ** 0.5
        * (1 + prec * (vals - mu) ** 2 / nu) ** ((-nu - 1) / 2)
    )

    return density


def main():
    mu = 0
    var = 1.0
    precision = 1 / np.sqrt(var)
    x_vals = np.linspace(mu - var * 4.0, mu + var * 4.0, num=250)
    high_nu_density = calculate_students_t_distribution(x_vals, mu, precision, 100.0)
    middle_nu_density = calculate_students_t_distribution(x_vals, mu, precision, 1.0)
    low_nu_density = calculate_students_t_distribution(x_vals, mu, precision, 0.1)

    plt.plot(x_vals, high_nu_density, label=r"$\nu=100.0$")
    plt.plot(x_vals, middle_nu_density, label=r"$\nu=1.0$")
    plt.plot(x_vals, low_nu_density, label=r"$\nu=0.1$")
    plt.legend()
    plt.title("Student's t-distribution")

    plt.figure()
    x_n = np.random.normal(loc=mu, scale=np.sqrt(var), size=100)
    # Insert outliers
    x_n[90:100] = 10
    x_vals = np.linspace(mu - 11.0, mu + 11.0, num=250)
    plt.hist(
        x=x_n,
        bins=50,
        range=(-11, 11),
        density=True,
        zorder=1,
    )
    mu = np.mean(x_n)
    var = np.var(x_n)
    precision = 1 / np.sqrt(var)
    stu_density = calculate_students_t_distribution(x_vals, mu, precision, 100.0)
    plt.plot(x_vals, stu_density, label=r"Student's t")
    gauss_density = draw_gaussian_distribution.calculate_gaussian_distribution(
        x_vals, mu, np.sqrt(var)
    )
    plt.plot(x_vals, gauss_density, label=r"Gaussian")
    plt.legend()
    plt.title("Student's t and Gaussian distribution with outliers")


if __name__ == "__main__":
    main()
    if len(sys.argv) == 2 and sys.argv[1] == "NotShow":
        print("It shows nothing")
    else:
        plt.show()
