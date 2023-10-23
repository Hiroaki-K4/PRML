import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def main():
    fig = plt.figure(figsize=(12, 8))

    print("Create beta distribution gif...")
    gif = FuncAnimation(fig, update, frames=len(mu_vals), interval=100)
    gif.save("images/beta_dist.gif")
    print("Finish!!")


if __name__ == "__main__":
    main()
