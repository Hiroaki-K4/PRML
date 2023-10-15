import sys

import matplotlib.pyplot as plt
import numpy as np


def get_axis_value_for_trigonometric_graph():
    axis_x = np.array([0.5, 0.0, 1.0])
    axis_y = np.array([0.5 * np.sqrt(3.0), 0.0, 0.0])
    axis_u = np.array([-0.5, 1.0, -0.5])
    axis_v = np.array([-0.5 * np.sqrt(3.0), 0.0, 0.5 * np.sqrt(3.0)])

    return axis_x, axis_y, axis_u, axis_v


def get_grid_value_for_trigonometric_graph():
    axis_vals = np.arange(start=0.0, stop=1.1, step=0.1)
    grid_x = np.hstack([0.5 * axis_vals, axis_vals, 0.5 * axis_vals + 0.5])
    grid_y = np.hstack(
        [
            0.5 * axis_vals * np.sqrt(3.0),
            np.zeros_like(axis_vals),
            0.5 * (1.0 - axis_vals) * np.sqrt(3.0),
        ]
    )
    grid_u = np.hstack([0.5 * axis_vals, 0.5 * (1.0 - axis_vals), -axis_vals])
    grid_v = np.hstack(
        [
            -0.5 * axis_vals * np.sqrt(3.0),
            0.5 * (1.0 - axis_vals) * np.sqrt(3.0),
            np.zeros_like(axis_vals),
        ]
    )

    return grid_x, grid_y, grid_u, grid_v, axis_vals


def draw_base_trigonometric_graph(
    axis_x, axis_y, axis_u, axis_v, grid_x, grid_y, grid_u, grid_v, axis_vals
):
    plt.figure(figsize=(10, 10))

    # Plot axis line
    plt.quiver(
        axis_x,
        axis_y,
        axis_u,
        axis_v,
        scale_units="xy",
        scale=1,
        units="dots",
        width=1.5,
        headwidth=1.5,
        fc="black",
        linestyle="-",
    )

    # Plot grid line
    plt.quiver(
        grid_x,
        grid_y,
        grid_u,
        grid_v,
        scale_units="xy",
        scale=1,
        units="dots",
        width=0.1,
        headwidth=0.1,
        fc="none",
        ec="gray",
        linewidth=1.5,
        linestyle=":",
    )

    for val in axis_vals:
        # \mu_0 axis scale
        plt.text(
            x=0.5 * val,
            y=0.5 * val * np.sqrt(3.0),
            s=str(np.round(1.0 - val, 1)) + " " * 2,
            ha="right",
            va="bottom",
            rotation=-60,
        )
        # \mu_1 axis scale
        plt.text(
            x=val,
            y=0.0,
            s=str(np.round(val, 1)) + " " * 10,
            ha="center",
            va="center",
            rotation=60,
        )
        # \mu_2 axis scale
        plt.text(
            x=0.5 * val + 0.5,
            y=0.5 * (1.0 - val) * np.sqrt(3.0),
            s=" " * 3 + str(np.round(1.0 - val, 1)),
            ha="left",
            va="center",
        )

    # Labels of each axis
    plt.text(
        x=0.25,
        y=0.25 * np.sqrt(3.0),
        s="$\mu_0$" + " " * 5,
        ha="right",
        va="center",
        size=25,
    )
    plt.text(x=0.5, y=0.0, s="\n" + "$\mu_1$", ha="center", va="top", size=25)
    plt.text(
        x=0.75,
        y=0.25 * np.sqrt(3.0),
        s=" " * 4 + "$\mu_2$",
        ha="left",
        va="center",
        size=25,
    )

    plt.xticks(ticks=[0.0, 0.5, 1.0], labels="")
    plt.yticks(ticks=[0.0, 0.25 * np.sqrt(3.0), 0.5 * np.sqrt(3.0)], labels="")
    plt.grid()
    plt.axis("equal")
    plt.suptitle(t="Trigonometric graph", fontsize=20)
    plt.title(label="$\mu=(\mu_0, \mu_1, \mu_2)$", loc="left")


if __name__ == "__main__":
    axis_x, axis_y, axis_u, axis_v = get_axis_value_for_trigonometric_graph()
    grid_x, grid_y, grid_u, grid_v, axis_vals = get_grid_value_for_trigonometric_graph()
    draw_base_trigonometric_graph(
        axis_x, axis_y, axis_u, axis_v, grid_x, grid_y, grid_u, grid_v, axis_vals
    )
    if len(sys.argv) == 2 and sys.argv[1] == "NotShow":
        print("It shows nothing")
    else:
        plt.show()
