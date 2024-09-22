import cv2
import numpy as np


def segment_image_by_kmeans(input_img, k):
    print(k)
    # Create random initial means
    u_k = np.random.randint(0, 256, size=(k, 3))

    # Optimize clusters and means
    # TODO Add optimization process
    # Create result image


def main(input_img_path):
    input_img = cv2.imread(input_img_path)
    print(input_img)
    k = 3
    segment_image_by_kmeans(input_img, k)


if __name__ == "__main__":
    input_img_path = "images/husky.jpg"
    main(input_img_path)
