import cv2
import os
import random
import numpy as np
from tqdm import tqdm


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
            # Invert pixels as noise with a 10% probability
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


def convert_binary_to_rgb(img):
    rgb_img = np.zeros((img.shape[0], img.shape[1], 3))
    for i in range(rgb_img.shape[0]):
        for j in range(rgb_img.shape[1]):
            if img[i][j] == 1:
                rgb_img[i][j] = np.array([0, 0, 0])
            elif img[i][j] == -1:
                rgb_img[i][j] = np.array([255, 255, 255])
            else:
                raise RuntimeError("The pixel value is wrong")

    return rgb_img


def calculate_multiplication_with_adjacent_pixels(denoise_img, row, col, denoise_val):
    adj_sum = 0
    for i in range(row - 1, row + 2):
        for j in range(col - 1, col + 2):
            if i == row and j == col:
                continue
            # Skip out of range of image
            if i < 0 or i >= denoise_img.shape[0] or j < 0 or j >= denoise_img.shape[1]:
                continue
            adj_sum += denoise_val * denoise_img[i][j]

    return adj_sum


def calculate_energy(x_sum, adj_sum, xy_sum, h, beta, eta):
    E = h * x_sum - beta * adj_sum - eta * xy_sum
    return E


def update_pixel(denoise_img, noise_img, target_row, target_col):
    # Parameters of energy function
    h, beta, eta = 0, 1.0, 2.1

    # Get energy of -1
    denoise_val = -1
    x_sum_0 = denoise_val
    adj_sum_0 = calculate_multiplication_with_adjacent_pixels(
        denoise_img, target_row, target_col, denoise_val
    )
    xy_sum_0 = denoise_val * noise_img[target_row][target_col]
    E_0 = calculate_energy(x_sum_0, adj_sum_0, xy_sum_0, h, beta, eta)

    # Get energy of 1
    denoise_val = 1
    x_sum_1 = denoise_val
    adj_sum_1 = calculate_multiplication_with_adjacent_pixels(
        denoise_img, target_row, target_col, denoise_val
    )
    xy_sum_1 = denoise_val * noise_img[target_row][target_col]
    E_1 = calculate_energy(x_sum_1, adj_sum_1, xy_sum_1, h, beta, eta)

    if E_0 < E_1:
        denoise_img[target_row][target_col] = -1
    else:
        denoise_img[target_row][target_col] = 1

    return denoise_img


def remove_noise_by_using_graphical_model(noise_img):
    denoise_img = np.copy(noise_img)

    for target_row in tqdm(range(noise_img.shape[0])):
        for target_col in range(noise_img.shape[1]):
            denoise_img = update_pixel(denoise_img, noise_img, target_row, target_col)

    return denoise_img


def main(test_img_path):
    ori_img = cv2.imread(test_img_path)
    # Align pixels(only (0,0,0) or (255,255,255))
    align_img = align_pixel_value(ori_img)

    # Create noise image
    noise_img = create_noise_image(align_img)
    noise_img_path = os.path.join(os.path.dirname(test_img_path), "noise.png")
    cv2.imwrite(noise_img_path, noise_img)

    # Convert rgb to binary(-1 or 1)
    bi_img = convert_rgb_to_binary(noise_img)

    # Remove noise
    denoise_img = remove_noise_by_using_graphical_model(bi_img)

    # Convert binary(-1 or 1) to rgb
    converted_denoise_img = convert_binary_to_rgb(denoise_img)
    denoise_img_path = os.path.join(os.path.dirname(test_img_path), "denoise.png")
    cv2.imwrite(denoise_img_path, converted_denoise_img)


if __name__ == "__main__":
    test_img_path = "images/test_data.png"
    main(test_img_path)
