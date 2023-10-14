import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def main():
    # 軸目盛の位置を指定
    axis_vals = np.arange(start=0.0, stop=1.1, step=0.1)

    # 軸線用の値を作成
    axis_x = np.array([0.5, 0.0, 1.0])
    axis_y = np.array([0.5 * np.sqrt(3.0), 0.0, 0.0])
    axis_u = np.array([-0.5, 1.0, -0.5])
    axis_v = np.array([-0.5 * np.sqrt(3.0), 0.0, 0.5 * np.sqrt(3.0)])

    # グリッド線用の値を作成
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
    print(grid_x)
    print(grid_y)
    print(grid_u)
    print(grid_v)
    input()


if __name__ == "__main__":
    main()
