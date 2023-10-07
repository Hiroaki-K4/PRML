import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

mu_vals = np.arange(start=0.0, stop=1.01, step=0.01)
N = 10
x_vals = np.arange(N + 1)


def calculate_binomial_probability(x_vals, N, mu):
    probs = []
    for i in range(x_vals.shape[0]):
        combi = math.factorial(N) / (math.factorial(N - i) * math.factorial(i))
        prob = combi * mu**i * (1 - mu) ** (N - i)
        probs.append(prob)

    return probs


def update(i):
    plt.cla()

    mu = mu_vals[i]
    prob = calculate_binomial_probability(x_vals, N, mu)

    # Plot binomial distribution
    plt.bar(x=x_vals, height=prob, color="blue")
    plt.xlabel("x")
    plt.ylabel("probability")
    plt.suptitle("Binomial distribution", fontsize=20)
    plt.title("$\mu=" + str(np.round(mu, 2)) + ", N=" + str(N) + "$", loc="left")
    plt.xticks(ticks=x_vals)
    plt.grid()
    plt.ylim(-0.1, 1.1)


def main():
    fig = plt.figure(figsize=(12, 8))

    print("Create binomial distribution gif...")
    gif = FuncAnimation(fig, update, frames=len(mu_vals), interval=100)
    gif.save("images/binomial_dist.gif")
    print("Finish!!")


if __name__ == "__main__":
    main()
