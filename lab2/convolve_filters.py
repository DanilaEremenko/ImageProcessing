import numpy as np
from matplotlib import pyplot as plt
from numba import jit


@jit(nopython=True)
def conv_py(arr, kernel):
    kernel_len = len(arr)
    sum = 0
    for x in range(kernel_len):
        for y in range(kernel_len):
            sum += arr[x, y] * kernel[x, y]
    return sum


@jit(nopython=True)
def conv_np(arr, kernel):
    return np.sum(np.multiply(arr, kernel))


def exapand_img(img_arr, kernel_shape):
    return np.pad(img_arr, kernel_shape[0] // 2, mode='reflect')


def full_adaptive_conv(img_arr, kernel_data, kernel_args, conv_func=conv_py):
    # expand
    exp_img = exapand_img(img_arr, kernel_data['shape'])

    # convolve
    res = np.zeros(img_arr.shape)
    for x in range(img_arr.shape[0]):
        for y in range(img_arr.shape[1]):
            kernel = kernel_data['func'](
                exp_img[x:x + kernel_data['shape'][0], y:y + kernel_data['shape'][0]],
                **kernel_args
            )

            res[x][y] = conv_func(
                arr=exp_img[x:x + kernel.shape[0], y:y + kernel.shape[1]],
                kernel=kernel
            )

    return res


def full_conv(img_arr, kernel, conv_func=conv_py):
    # expand
    exp_img = exapand_img(img_arr, kernel.shape)

    # convolve
    res = np.zeros(img_arr.shape)
    for x in range(img_arr.shape[0]):
        for y in range(img_arr.shape[1]):
            res[x][y] = conv_func(
                arr=exp_img[x:x + kernel.shape[0], y:y + kernel.shape[1]],
                kernel=kernel
            )

    return res


def calculate_diff(orig_img, comp_img):
    return np.mean((orig_img - comp_img) ** 2)


def draw_image(img_arr, title):
    # plot results
    if title is not None:
        plt.imshow(img_arr, cmap='gray')
        plt.title(title)
        plt.show()
