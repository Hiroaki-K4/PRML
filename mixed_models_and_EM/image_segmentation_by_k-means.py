import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from tqdm import tqdm


def create_segmented_image(result_label, cluster_colors):
    # Create an empty image to store the RGB segmented image
    segmented_img = np.zeros((*result_label.shape, 3), dtype=np.uint8)

    # Assign each segment its corresponding color
    for i in range(len(cluster_colors)):
        segmented_img[result_label == i] = cluster_colors[i]

    return segmented_img


def segment_image_by_kmeans(input_img, k):
    # Create random initial means (cluster centroids)
    u_k = np.random.randint(0, 256, size=(k, 3))

    # Reshape the input image to a 2D array where each row is a pixel (R, G, B)
    pixels = input_img.reshape(-1, 3)

    # Prepare an array to store the assigned cluster for each pixel
    class_arr = np.zeros(pixels.shape[0], dtype=int)

    # Iterate until convergence (in this version, just perform one iteration)
    for _ in range(10):
        # Compute the squared Euclidean distance from each pixel to each cluster center
        distances = np.sqrt(((pixels[:, np.newaxis] - u_k) ** 2).sum(axis=2))

        # Assign each pixel to the nearest cluster
        class_arr = np.argmin(distances, axis=1)

        old_u_k = np.copy(u_k)
        # Recalculate the new means for each cluster
        for i in range(k):
            if np.any(class_arr == i):
                u_k[i] = pixels[class_arr == i].mean(axis=0)

        if np.array_equal(old_u_k, u_k):
            break

    # Reshape the class_arr back to the original image shape
    result_label = class_arr.reshape(input_img.shape[:2])

    # Create result image
    segmented_img = create_segmented_image(result_label, u_k)

    return segmented_img


def save_slideshow_as_gif(images, gif_path):
    fig, ax = plt.subplots()
    ax.axis("off")

    # Display initial image
    img_display = ax.imshow(images[0])

    # Update function
    def update(frame):
        img_display.set_data(images[frame])
        return [img_display]

    # Create animation
    ani = FuncAnimation(fig, update, frames=len(images), interval=1000, repeat=True)

    ani.save(gif_path, writer="imagemagick", fps=1)
    plt.close()


def main(input_img_path, ks):
    np.random.seed(314)
    input_img = cv2.imread(input_img_path)
    # Prepare for animation
    original_img = cv2.putText(
        np.copy(input_img),
        "Original image",
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 0, 255),
        3,
        cv2.LINE_AA,
    )
    images = [original_img[..., ::-1]]

    for k in tqdm(ks):
        segmented_img = segment_image_by_kmeans(input_img, k)
        segmented_img = cv2.putText(
            segmented_img,
            "k={0}".format(k),
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )
        images.append(segmented_img[..., ::-1])
        segmented_img_path = os.path.join(
            os.path.dirname(input_img_path), "segmented_{0}.png".format(k)
        )
        cv2.imwrite(segmented_img_path, segmented_img)

    gif_path = os.path.join(os.path.dirname(input_img_path), "output.gif")
    save_slideshow_as_gif(images, gif_path)


if __name__ == "__main__":
    input_img_path = "images/husky.jpg"
    ks = [5, 10, 15, 20]
    main(input_img_path, ks)
