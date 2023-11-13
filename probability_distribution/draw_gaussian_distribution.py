import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def calculate_gaussian_distribution(x_vals, mu, sigma):
    density = (
        1
        / (2 * np.pi * sigma**2) ** 0.5
        * np.exp(-1 / (2 * sigma**2) * (x_vals - mu) ** 2)
    )

    return density


mu = 1.0  # Mean
sigma = 2.5  # Standard deviation
N = 1000  # Data number
x_n = np.random.normal(loc=mu, scale=sigma, size=N)
x_vals = np.linspace(mu - sigma * 4.0, mu + sigma * 4.0, num=250)
density = calculate_gaussian_distribution(x_vals, mu, sigma)


def update(n):
    plt.cla()
    plt.hist(
        x=x_n[: (n + 1)],
        bins=50,
        range=(x_vals.min(), x_vals.max()),
        density=True,
        zorder=1,
    )
    plt.plot(x_vals, density, color="orange", linestyle="--", zorder=2)
    plt.scatter(x=x_n[n], y=0.0, color="orange", s=100, zorder=3)
    plt.xlabel("x")
    plt.ylabel("density")
    plt.suptitle("Gaussian Distribution", fontsize=20)
    plt.title(
        "$\mu=" + str(mu) + ", \sigma=" + str(sigma) + ", N=" + str(n + 1) + "$",
        loc="left",
    )
    plt.grid()
    plt.ylim(ymin=-0.01, ymax=density.max() + 0.1)


if __name__ == "__main__":
    fig = plt.figure(figsize=(8, 6))
    fig.suptitle("Gaussian Distribution", fontsize=20)

    print("Create Gaussian distribution gif...")
    gif = FuncAnimation(fig, update, frames=250, interval=100)
    gif.save("images/gaussian_dist.gif")
    print("Finish!!")
