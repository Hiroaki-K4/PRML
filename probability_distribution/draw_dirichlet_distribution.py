import draw_trigonometric_graph
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.special import gamma

# Parameter to be changed
alpha_1_vals = np.arange(start=1.0, stop=10.1, step=0.1).round(decimals=1)
# Parameters to be fixed
alpha_2 = 2.0
alpha_3 = 3.0
frame_num = len(alpha_1_vals)

fig = plt.figure(figsize=(12, 10), facecolor="white")
ax = fig.add_subplot(projection="3d")
fig.suptitle(t="Dirichlet distribution", fontsize=20)

dens_min = 0.0
dens_max = 25.0
dens_levels = np.linspace(dens_min, dens_max, num=11)

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


(
    x_points,
    y_0_grid,
    y_1_grid,
    y_shape,
) = calculate_3d_coords_for_contour_map()


def calculate_probability_density(mu_k, alpha_k):
    multi = 1
    for alpha in alpha_k:
        multi *= gamma(alpha)

    norm = gamma(sum(alpha_k)) / multi

    mus = 1
    for i in range(len(mu_k)):
        mus *= mu_k[i] ** (alpha_k[i] - 1)

    prob = norm * mus

    return prob


def update(i):
    plt.cla()

    alpha_1 = alpha_1_vals[i]
    alpha_v = np.array([alpha_1, alpha_2, alpha_3])
    dens_vals = np.array(
        [
            calculate_probability_density(mu_v, alpha_v)
            if all(mu_v != np.nan)
            else np.nan
            for mu_v in x_points
        ]
    )

    # Plot grid
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
    # Plot axis
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

    # Plot axis scale
    for val in axis_vals:
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
        s="$\mu_1$",
        ha="right",
        va="center",
        size=25,
    )
    ax.text(
        x=0.5, y=0.0 - 0.1, z=0.0 - 0.1, s="$\mu_2$", ha="center", va="top", size=25
    )
    ax.text(
        x=0.75 + 0.1,
        y=0.25 * np.sqrt(3.0),
        z=0.0,
        s="$\mu_3$",
        ha="left",
        va="center",
        size=25,
    )
    ax.contour(
        y_0_grid,
        y_1_grid,
        dens_vals.reshape(y_shape),
        vmin=dens_min,
        vmax=dens_max,
        levels=dens_levels,
        offset=0.0,
    )
    ax.plot_surface(
        y_0_grid, y_1_grid, dens_vals.reshape(y_shape), cmap="viridis", alpha=0.8
    )
    ax.set_xticks(ticks=[0.0, 0.5, 1.0], labels="")
    ax.set_yticks(ticks=[0.0, 0.25 * np.sqrt(3.0), 0.5 * np.sqrt(3.0)], labels="")
    ax.set_zlabel(zlabel="density")
    ax.set_zlim(bottom=dens_min, top=dens_max)
    ax.set_box_aspect(aspect=(1, 1, 1))
    param_text = "$\\alpha=(" + ", ".join([str(alpha) for alpha in alpha_v]) + ")$"
    ax.set_title(label=param_text, loc="left")


def main():
    print("Create dirichlet distribution gif...")
    ani = FuncAnimation(fig=fig, func=update, frames=frame_num, interval=100)
    ani.save("images/dirichlet_dist.gif")
    print("Finish!!")


if __name__ == "__main__":
    main()
