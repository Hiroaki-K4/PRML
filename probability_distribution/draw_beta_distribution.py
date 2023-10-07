import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.special import gamma

mu_vals = np.arange(start=0.0, stop=1.01, step=0.01)
a = 0
b = 0


def calculate_mu_distribution(a, b):
    probs = []
    for mu in mu_vals:
        ganma = gamma(a + b) / (gamma(a) * gamma(b))
        prob = ganma * mu ** (a - 1) * (1 - mu) ** (b - 1)
        probs.append(prob)

    return probs


def update(i):
    bin_val = np.random.randint(2)
    global a, b
    if bin_val == 1:
        a += 1
    else:
        b += 1

    probs = calculate_mu_distribution(a, b)
    plt.plot(mu_vals, probs)
    plt.xlabel("$\mu$")
    plt.ylabel("probability")
    plt.suptitle("Beta distribution", fontsize=20)
    plt.title("a=" + str(a) + ", b=" + str(b), loc="left")
    plt.grid()


def main():
    fig = plt.figure(figsize=(12, 8))

    print("Create beta distribution gif...")
    gif = FuncAnimation(fig, update, frames=len(mu_vals), interval=100)
    gif.save("images/beta_dist.gif")
    print("Finish!!")


if __name__ == "__main__":
    main()
