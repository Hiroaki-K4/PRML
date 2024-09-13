import cv2
import os
import random
import numpy as np


def align_pixel_value(ori_img):
    align_img = np.copy(ori_img)
    for i in range(align_img.shape[0]):
        for j in range(align_img.shape[1]):
            if np.sum(ori_img[i][j]) > 0:
                align_img[i][j] = np.array([255, 255, 255])

    return align_img


def create_noise_image(ori_img):
    noise_img = np.copy(ori_img)
    for i in range(noise_img.shape[0]):
        for j in range(noise_img.shape[1]):
            if random.randint(1, 100) <= 10:
                inv_arr = np.abs(ori_img[i][j] - np.array([255, 255, 255]))
                noise_img[i][j] = inv_arr

    return noise_img


def convert_rgb_to_binary(img):
    bi_img = np.zeros((img.shape[0], img.shape[1]))
    for i in range(bi_img.shape[0]):
        for j in range(bi_img.shape[1]):
            if np.array_equal(img[i][j], np.array([0, 0, 0])):
                bi_img[i][j] = 1
            elif np.array_equal(img[i][j], np.array([255, 255, 255])):
                bi_img[i][j] = -1
            else:
                raise RuntimeError("The pixel value is wrong")

    return bi_img


# def remove_noise_by_using_graphical_model(noise_img):


def main(test_img_path):
    ori_img = cv2.imread(test_img_path)
    align_img = align_pixel_value(ori_img)

    # Create noise image
    noise_img = create_noise_image(align_img)
    noise_img_path = os.path.join(os.path.dirname(test_img_path), "noise.png")
    cv2.imwrite(noise_img_path, noise_img)

    # Convert rgb to binary(-1 or 1)
    bi_img = convert_rgb_to_binary(noise_img)

    # Remove noise
    # TODO Add noise reduction
    # remove_noise_by_using_graphical_model(noise_img)


if __name__ == "__main__":
    test_img_path = "images/test_data.png"
    main(test_img_path)
