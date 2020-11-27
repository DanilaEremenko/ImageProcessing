import numpy as np
from matplotlib import pyplot as plt


def conv(img_sub_arr, kernel):
    return np.sum(np.multiply(img_sub_arr, kernel))


def full_conv(img_arr, kernel):
    # expand
    if len(kernel) > 2:

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
            res[x][y] = conv(
                img_sub_arr=expanded_arr[x:x + kernel.shape[0], y:y + kernel.shape[1]],
                kernel=kernel
            )

    # cut if was expanded
    if expand_len != 0:
        res = res[expand_len:-expand_len, expand_len:-expand_len]

    return res


def convolve_and_show(img_arr, kernel=None, title=None):
    if kernel is not None:
        res = full_conv(img_arr, kernel)
    else:
        res = img_arr

    if title is not None:
        plt.imshow(res, cmap='gray')

        plt.title(title)
        plt.show()
