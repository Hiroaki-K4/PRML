import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.special import gamma

mu = 0.75  # Mean
var = 1.0  # Variance
N = 1000  # Data number
x_n = np.random.normal(loc=mu, scale=np.sqrt(var), size=N)
var_vals = np.linspace(0, 2, num=250)
a_0 = 1  # Init gamma parameter
b_0 = 1  # Init gamma parameter


def calculate_beta_distribution(vals, a, b):
    density = []
    for val in vals:
        res = 1 / gamma(a) * b**a * val ** (a - 1) * np.exp(-b * val)
        density.append(res)

    return density


def update(n):
    plt.cla()
    if n == 0:
        var_ml = 0
    else:
        var_ml = np.mean((x_n[0:n] - mu) ** 2)

    a_n = a_0 + n / 2
    b_n = b_0 + n / 2 * var_ml
    density = calculate_beta_distribution(var_vals, a_n, b_n)

    plt.plot(var_vals, density, zorder=2)
    plt.xlabel("$\lambda$")
    plt.ylabel("density")
    plt.suptitle("Gaussian distribution with known mean", fontsize=20)
    plt.title(
        "$a="
        + str(round(a_n, 1))
        + ", b="
        + str(round(b_n, 1))
        + ", N="
        + str(n)
        + "$",
        loc="left",
    )
    plt.grid()
    plt.ylim(ymin=-0.01, ymax=max(density) + 0.1)


if __name__ == "__main__":
    fig = plt.figure(figsize=(8, 6))
    fig.suptitle("Gaussian Distribution", fontsize=20)

    print("Create a gif of Gaussian distribution with known mean...")
    gif = FuncAnimation(fig, update, frames=30, interval=150)
    gif.save("images/gaussian_dist_with_known_mean.gif")
    print("Finish!!")
