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


def full_conv(img_arr, kernel_data, conv_func=conv_py):
    # adaptive kernels mode
    if type(kernel_data) is dict:
        adaptive_mode = True
        kernel = np.zeros(kernel_data['shape'])
    # default mode
    elif type(kernel_data) is np.ndarray:
        adaptive_mode = False
        kernel = kernel_data
    else:
        raise Exception('Unexpected kernel data ' + str(type(kernel_data)))

    # expand
    if len(kernel) > 2:
        if not all([dim % 2 == 1 for dim in kernel.shape]):
            raise Exception('All kernel dims should be an odd')

        expand_len = int(len(kernel) // 2)
        expanded_arr = img_arr

        for i in range(expand_len):
            expanded_arr = np.concatenate((expanded_arr[0][np.newaxis], expanded_arr), axis=0)
            expanded_arr = np.concatenate((expanded_arr[:, 0][:, np.newaxis], expanded_arr), axis=1)
            expanded_arr = np.concatenate((expanded_arr, expanded_arr[-1][np.newaxis]), axis=0)
            expanded_arr = np.concatenate((expanded_arr, expanded_arr[:, -1][:, np.newaxis]), axis=1)

    else:
        expand_len = 0
        expanded_arr = img_arr

    # convolve
    res = np.zeros(img_arr.shape)
    for x in range(0, expanded_arr.shape[0] - kernel.shape[0] + 1):
        for y in range(0, expanded_arr.shape[1] - kernel.shape[1] + 1):
            if adaptive_mode:
                kernel = kernel_data['func'](
                    expanded_arr[x:x + kernel.shape[0], y:y + kernel.shape[1]],
                    **kernel_data['args']
                )

            res[x][y] = conv_func(
                arr=expanded_arr[x:x + kernel.shape[0], y:y + kernel.shape[1]],
                kernel=kernel
            )

    return res


def convolve_and_show(img_arr, kernel=None, title=None, original_image=None):
    # make convolution
    if kernel is not None:
        res = full_conv(img_arr, kernel)
    else:
        res = img_arr

    # calculate difference between images
    if original_image is not None:
        diff = np.sum(abs(original_image - res)) \
               / np.sum((np.ones(shape=res.shape) * 255))
    else:
        diff = None

    # plot results
    if title is not None:
        plt.imshow(res, cmap='gray')
        if diff is None:
            plt.title(title)
        else:
            plt.title("%s\n diff = %.3f" % (title, diff) + '%')
        plt.show()
