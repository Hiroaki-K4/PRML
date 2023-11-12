import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

import draw_gaussian_distribution

mu = 0.75  # Average
var = 0.1  # Standard deviation
N = 1000  # Data number
x_n = np.random.normal(loc=mu, scale=np.sqrt(var), size=N)
x_vals = np.linspace(-1, 1, num=250)
pre_mu = 0  # prior distribution
pre_var = var  # prior distribution


def update(n):
    plt.cla()
    if n == 0:
        mu_ml = 0
    else:
        mu_ml = np.average(x_n[0:n])

    mu_n = (
        var / (n * pre_var + var) * pre_mu + (n * pre_var) / (n * pre_var + var) * mu_ml
    )
    var_inv = 1 / pre_var + n / var
    var_n = var_inv ** (-1)
    density = draw_gaussian_distribution.calculate_gaussian_distribution(
        x_vals, mu_n, np.sqrt(var_n)
    )

    plt.plot(x_vals, density, zorder=2)
    plt.xlabel("$\mu$")
    plt.ylabel("density")
    plt.suptitle("Gaussian distribution with known variance", fontsize=20)
    plt.title(
        "$\mu_N="
        + str(round(mu_n, 2))
        + ", \sigma_N="
        + str(round(np.sqrt(var_n), 2))
        + ", N="
        + str(n)
        + "$",
        loc="left",
    )
    plt.grid()
    plt.ylim(ymin=-0.01, ymax=density.max() + 0.1)


if __name__ == "__main__":
    fig = plt.figure(figsize=(8, 6))
    fig.suptitle("Gaussian Distribution", fontsize=20)

    print("Create a gif of Gaussian distribution with known variance...")
    gif = FuncAnimation(fig, update, frames=25, interval=250)
    gif.save("images/gaussian_dist_with_known_var.gif")
    print("Finish!!")
