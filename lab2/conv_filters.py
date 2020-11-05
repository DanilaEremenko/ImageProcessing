from matplotlib import pyplot as plt
import cv2
import numpy as np
from scipy.ndimage.filters import convolve


def convolve_and_show(img, kernel=None, title=None):
    if kernel is not None:
        res = convolve(img, kernel, mode='constant', cval=0.0)
    else:
        res = img

    if title is not None:
        plt.imshow(res)
        plt.title(title)
        plt.show()


def main():
    img = cv2.cvtColor(cv2.imread('images/vert.jpg'), cv2.COLOR_BGR2GRAY)

    convolve_and_show(img, title='ORIGINAL IMAGE')

    # ----------------------- ROBERTS --------------------------------------
    convolve_and_show(
        img=img,
        kernel=np.array([
            [1, 0],
            [0, -1]]),
        title='ROBERTS LEFT'
    )

    convolve_and_show(
        img=img,
        kernel=np.array([
            [1, 0],
            [0, -1]]),
        title='ROBERTS RIGHT'
    )

    # ----------------------- SOBEL --------------------------------------
    convolve_and_show(
        img=img,
        kernel=np.array([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ]),
        title='SOBEL VERTICAL'
    )

    convolve_and_show(
        img=img,
        kernel=np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [1, 0, 1]
        ]),
        title='SOBEL HORIZONTAL'
    )

    # ----------------------- PREWITT--------------------------------------
    convolve_and_show(
        img=img,
        kernel=np.array([
            [-1, -1, -1],
            [0, 0, 0],
            [1, 1, 1]
        ]),
        title='PREWITT HORIZONTAL'
    )

    convolve_and_show(
        img=img,
        kernel=np.array([
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]
        ]),
        title='PREWITT VERTICAL'
    )


if __name__ == '__main__':
    main()
