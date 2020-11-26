from matplotlib import pyplot as plt
import cv2
import numpy as np


def conv(img_sub_arr, kernel):
    return np.sum(np.multiply(img_sub_arr, kernel))


def full_conv(img_arr, kernel):
    res = np.zeros(img_arr.shape)
    for x in range(0, img_arr.shape[0] - kernel.shape[0] + 1):
        for y in range(0, img_arr.shape[1] - kernel.shape[1] + 1):
            res[x][y] = conv(
                img_sub_arr=img_arr[x:x + kernel.shape[0], y:y + kernel.shape[1]],
                kernel=kernel
            )
    return res


def convolve_and_show(img_arr, kernel=None, title=None):
    if kernel is not None:
        if len(kernel) > 2:
            expanded_arr = img_arr.copy()

            expanded_arr = np.concatenate((expanded_arr[0][np.newaxis], expanded_arr), axis=0)
            expanded_arr = np.concatenate((expanded_arr[:, 0][:, np.newaxis], expanded_arr), axis=1)
            expanded_arr = np.concatenate((expanded_arr, expanded_arr[-1][np.newaxis]), axis=0)
            expanded_arr = np.concatenate((expanded_arr, expanded_arr[:, -1][:, np.newaxis]), axis=1)

            expanded_arr = np.array(expanded_arr, dtype='uint8')

            res = full_conv(expanded_arr, kernel)
            res = res[1:-1, 1:-1]  # remove expanded part
        else:
            res = full_conv(img_arr, kernel)

        res = np.array(res)
    else:
        res = img_arr

    if title is not None:
        plt.imshow(res)
        plt.title(title)
        plt.show()


def main():
    img = cv2.cvtColor(cv2.imread('images/boundaries_yum.jpg'), cv2.COLOR_BGR2GRAY)

    convolve_and_show(img, title='ORIGINAL IMAGE')

    kernels_dict = {

        'ROBERTS LEFT': np.array([[1, 0],
                                  [0, -1]]),

        'ROBERTS RIGHT': np.array([[1, 0],
                                   [0, -1]]),

        'SOBEL HORIZONTAL': np.array([[-1, -2, -1],
                                      [0, 0, 0],
                                      [1, 2, 1]]),

        'SOBEL VERTICAL': np.array([[-1, 0, 1],
                                    [-2, 0, 2],
                                    [1, 0, 1]]),

        'PREWITT HORIZONTAL': np.array([[-1, -1, -1],
                                        [0, 0, 0],
                                        [1, 1, 1]]),

        'PREWITT VERTICAL': np.array([[-1, 0, 1],
                                      [-1, 0, 1],
                                      [-1, 0, 1]]),

    }

    # ----------------------- RUNING --------------------------------------
    for title, kernel in kernels_dict.items():
        convolve_and_show(
            img_arr=img,
            kernel=kernel,
            title=title
        )


if __name__ == '__main__':
    main()
