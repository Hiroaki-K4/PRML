import sys

import draw_dirichlet_distribution
import draw_trigonometric_graph
import matplotlib.pyplot as plt
import numpy as np


def calculate_3d_coords_for_contour_map():
    y_0_vals = np.linspace(start=0.0, stop=1.0, num=201)
    y_1_vals = np.linspace(start=0.0, stop=0.5 * np.sqrt(3.0), num=200)

    # Create a grid of 2D coordinates
    y_0_grid, y_1_grid = np.meshgrid(y_0_vals, y_1_vals)
    y_shape = y_0_grid.shape

    # Convert to 3D coordinate values
    x_1_vals = y_0_grid.flatten() - y_1_grid.flatten() / np.sqrt(3.0)
    x_2_vals = 2.0 * y_1_grid.flatten() / np.sqrt(3.0)

    # Replace out-of-range points with missing values
    x_1_vals = np.where((x_1_vals >= 0.0) & (x_1_vals <= 1.0), x_1_vals, np.nan)
    x_2_vals = np.where((x_2_vals >= 0.0) & (x_2_vals <= 1.0), x_2_vals, np.nan)
    x_0_vals = 1.0 - x_1_vals - x_2_vals
    x_0_vals = np.where((x_0_vals >= 0.0) & (x_0_vals <= 1.0), x_0_vals, np.nan)
    x_points = np.stack([x_0_vals, x_1_vals, x_2_vals], axis=1)

    return x_points, y_0_grid, y_1_grid, y_shape


def draw_contour_map(
    axis_x,
    axis_y,
    axis_u,
    axis_v,
    grid_x,
    grid_y,
    grid_u,
    grid_v,
    axis_vals,
    x_points,
    y_0_grid,
    y_1_grid,
    y_shape,
):
    # Parameters for dirchlet distribution
    alpha_k = np.array([2.5, 3.5, 4.5])

    # Calculate probability density of dirchlet distribution
    dens_vals = np.array(
        [
            draw_dirichlet_distribution.calculate_probability_density(x_k, alpha_k)
            if all(x_k != np.nan)
            else np.nan
            for x_k in x_points
        ]
    )

    plt.figure(figsize=(12, 10), facecolor="white")
    # Plot grid
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
    # Plot axis
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

    for val in axis_vals:
        # Plot axis scale
        plt.text(
            x=0.5 * val,
            y=0.5 * val * np.sqrt(3.0),
            s=str(np.round(1.0 - val, 1)) + " " * 2,
            ha="right",
            va="bottom",
            rotation=-60,
        )
        plt.text(
            x=val,
            y=0.0,
            s=str(np.round(val, 1)) + " " * 10,
            ha="center",
            va="center",
            rotation=60,
        )
        plt.text(
            x=0.5 * val + 0.5,
            y=0.5 * (1.0 - val) * np.sqrt(3.0),
            s=" " * 3 + str(np.round(1.0 - val, 1)),
            ha="left",
            va="center",
        )

    plt.text(
        x=0.25,
        y=0.25 * np.sqrt(3.0),
        s="$x_0$" + " " * 5,
        ha="right",
        va="center",
        size=25,
    )
    plt.text(x=0.5, y=0.0, s="\n" + "$x_1$", ha="center", va="top", size=25)
    plt.text(
        x=0.75,
        y=0.25 * np.sqrt(3.0),
        s=" " * 4 + "$x_2$",
        ha="left",
        va="center",
        size=25,
    )
    cnf = plt.contourf(y_0_grid, y_1_grid, dens_vals.reshape(y_shape), alpha=0.8)

    plt.xticks(ticks=[0.0, 0.5, 1.0], labels="")
    plt.yticks(ticks=[0.0, 0.25 * np.sqrt(3.0), 0.5 * np.sqrt(3.0)], labels="")
    plt.grid()
    plt.axis("equal")
    plt.colorbar(cnf, label="density")
    plt.suptitle(t="Contour map", fontsize=20)
    param_text = (
        "$"
        + "\\alpha=("
        + ", ".join([str(val) for val in alpha_k])
        + ")"
        + ", \mu=(\mu_0, \mu_1, \mu_2)"
        + "$"
    )
    plt.title(label=param_text, loc="left")


if __name__ == "__main__":
    (
        axis_x,
        axis_y,
        axis_u,
        axis_v,
    ) = draw_trigonometric_graph.get_axis_value_for_trigonometric_graph()
    (
        grid_x,
        grid_y,
        grid_u,
        grid_v,
        axis_vals,
    ) = draw_trigonometric_graph.get_grid_value_for_trigonometric_graph()
    (
        x_points,
        y_0_grid,
        y_1_grid,
        y_shape,
    ) = draw_dirichlet_distribution.calculate_3d_coords_for_contour_map()
    draw_contour_map(
        axis_x,
        axis_y,
        axis_u,
        axis_v,
        grid_x,
        grid_y,
        grid_u,
        grid_v,
        axis_vals,
        x_points,
        y_0_grid,
        y_1_grid,
        y_shape,
    )
    if len(sys.argv) == 2 and sys.argv[1] == "NotShow":
        print("It shows nothing")
    else:
        plt.show()
