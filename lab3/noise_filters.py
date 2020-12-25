import cv2
import numpy as np
from my_image_processing import convolve_and_show
from noise_filters_cython import get_bilateral_kernel_cython


def get_gaussian_noise(mean, sigma, shape):
    return np.random.normal(mean, sigma, shape)


######################################################################
# -------------------- BILATERAL NP ----------------------------------
######################################################################
def gaussian_np(x, sigma):
    return (1.0 / (2 * np.pi * (sigma ** 2))) * np.exp(-(x ** 2) / (2 * (sigma ** 2)))


def distance_np(x1, y1, x2, y2):
    return np.sqrt(np.abs((x1 - x2) ** 2 - (y1 - y2) ** 2))


def get_bilateral_kernel_np(small_img_arr, sigma_i, sigma_s):
    # assert len(small_img_arr.shape) == 2
    # assert small_img_arr.shape[0] == small_img_arr.shape[1]
    kernel_len = len(small_img_arr)
    x_arr = np.zeros(shape=small_img_arr.shape)
    y_arr = np.zeros(shape=small_img_arr.shape)
    for x in range(kernel_len):
        for y in range(kernel_len):
            x_arr[x, y] = x
            y_arr[x, y] = y
    center = small_img_arr.shape[0] // 2
    gi = gaussian_np(small_img_arr - small_img_arr[center, center], sigma_i)
    gs = gaussian_np(distance_np(x_arr, y_arr, center, center), sigma_s)
    kernel = gi * gs
    return kernel / kernel.sum()


######################################################################
# -------------------- BILATERAL PY ----------------------------------
######################################################################

import math


def gaussian_py(x, sigma):
    return (1.0 / (2 * math.pi * (sigma ** 2))) * math.exp(-(x ** 2) / (2 * (sigma ** 2)))


def distance_py(x1, y1, x2, y2):
    return math.sqrt(abs((x1 - x2) ** 2 - (y1 - y2) ** 2))


def get_bilateral_kernel_py(small_img_arr, sigma_i, sigma_s):
    center = small_img_arr.shape[0] // 2
    kernel = np.zeros(shape=small_img_arr.shape)

    for x in range(len(small_img_arr)):
        for y in range(len(small_img_arr)):
            gi = gaussian_py(small_img_arr[x, y] - small_img_arr[center, center], sigma_i)
            gs = gaussian_py(distance_py(x, y, center, center), sigma_s)
            kernel[x, y] = gi * gs
    return kernel / kernel.sum()


######################################################################
# -------------------- GAUSSIAN --------------------------------------
######################################################################

def get_gaussian_kernel(kernel_len, sigma):
    center = kernel_len // 2
    kernel = np.zeros((kernel_len, kernel_len))
    for x in range(kernel_len):
        for y in range(kernel_len):
            diff = distance_py(x, y, center, center)
            kernel[x, y] = gaussian_py(diff, sigma=sigma)
    return kernel / np.sum(kernel)


def main():
    img = cv2.cvtColor(cv2.imread('images/test_dog.jpg'), cv2.COLOR_BGR2GRAY)

    sigma = int(0.02 * len(np.diag(img)))

    kernel_len = 5
    # kernel_len = np.int(np.sqrt(sigma) * 3)

    img_noised = img + get_gaussian_noise(mean=0, sigma=sigma, shape=img.shape)

    kernels_dict = {
        'MEAN FILTER': np.ones(shape=(kernel_len, kernel_len)) / (kernel_len * kernel_len),

        'GAUSSIAN FILTER': get_gaussian_kernel(kernel_len, sigma),

        'PYTHON BILATERAL FILTER': {
            'func': get_bilateral_kernel_np,
            'shape': (kernel_len, kernel_len),
            'args': {'sigma_i': sigma, 'sigma_s': sigma}
        },
    }

    # ----------------------- RUNING --------------------------------------
    convolve_and_show(img, title='ORIGINAL IMAGE')
    convolve_and_show(img_noised, title='NOISED IMAGE')
    import time
    for title, kernel in kernels_dict.items():
        start_time = time.time()
        convolve_and_show(
            img_arr=img_noised,
            kernel=kernel,
            title=title + '\n'
        )
        print(f"{title} time = {time.time() - start_time}")


if __name__ == '__main__':
    main()
