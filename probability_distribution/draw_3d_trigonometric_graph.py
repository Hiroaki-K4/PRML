import sys

import matplotlib.pyplot as plt
import numpy as np

import draw_contour_map
import draw_dirichlet_distribution
import draw_trigonometric_graph


def draw_3d_trigonometric_graph(
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

    fig = plt.figure(figsize=(12, 10), facecolor="white")
    ax = fig.add_subplot(projection="3d")
    ax.quiver(
        grid_x,
        grid_y,
        np.zeros_like(grid_x),
        grid_u,
        grid_v,
        np.zeros_like(grid_x),
        arrow_length_ratio=0.0,
        ec="gray",
        linewidth=1.5,
        linestyle=":",
    )
    ax.quiver(
        axis_x,
        axis_y,
        np.zeros_like(axis_x),
        axis_u,
        axis_v,
        np.zeros_like(axis_x),
        arrow_length_ratio=0.0,
        ec="black",
        linestyle="-",
    )

    for val in axis_vals:
        # Plot axis scale
        ax.text(
            x=0.5 * val - 0.05,
            y=0.5 * val * np.sqrt(3.0),
            z=0.0,
            s=str(np.round(1.0 - val, 1)),
            ha="center",
            va="center",
        )
        ax.text(
            x=val,
            y=0.0 - 0.05,
            z=0.0,
            s=str(np.round(val, 1)),
            ha="center",
            va="center",
        )
        ax.text(
            x=0.5 * val + 0.5 + 0.05,
            y=0.5 * (1.0 - val) * np.sqrt(3.0),
            z=0.0,
            s=str(np.round(1.0 - val, 1)),
            ha="center",
            va="center",
        )

    ax.text(
        x=0.25 - 0.1,
        y=0.25 * np.sqrt(3.0),
        z=0.0,
        s="$x_0$",
        ha="right",
        va="center",
        size=25,
    )
    ax.text(x=0.5, y=0.0 - 0.1, z=0.0 - 0.1, s="$x_1$", ha="center", va="top", size=25)
    ax.text(
        x=0.75 + 0.1,
        y=0.25 * np.sqrt(3.0),
        z=0.0,
        s="$x_2$",
        ha="left",
        va="center",
        size=25,
    )
    ax.contour(y_0_grid, y_1_grid, dens_vals.reshape(y_shape), offset=0.0)
    ax.plot_surface(
        y_0_grid, y_1_grid, dens_vals.reshape(y_shape), cmap="viridis", alpha=0.8
    )
    ax.set_xticks(ticks=[0.0, 0.5, 1.0], labels="")
    ax.set_yticks(ticks=[0.0, 0.25 * np.sqrt(3.0), 0.5 * np.sqrt(3.0)], labels="")
    ax.set_zlabel(zlabel="density")
    ax.set_box_aspect(aspect=(1, 1, 1))
    fig.suptitle(t="3D Triangraph", fontsize=20)
    param_text = (
        "$"
        + "\\alpha=("
        + ", ".join([str(val) for val in alpha_k])
        + ")"
        + ", \mu=(\mu_0, \mu_1, \mu_2)"
        + "$"
    )
    ax.set_title(label=param_text, loc="left")


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
    ) = draw_contour_map.calculate_3d_coords_for_contour_map()
    draw_3d_trigonometric_graph(
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
