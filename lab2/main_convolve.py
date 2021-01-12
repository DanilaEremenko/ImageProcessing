import cv2
import numpy as np
from convolve_filters import draw_image, full_conv


def main():
    img = cv2.cvtColor(cv2.imread('../dimages/boundaries_yum.jpg'), cv2.COLOR_BGR2GRAY)

    kernels_dict = {

        'ROBERTS LEFT': {
            'kernel': np.array([[1, 0],
                                [0, -1]]),
        },
        'ROBERTS RIGHT': {
            'kernel': np.array([[0, 1],
                                [-1, 0]]),
        },
        'SOBEL HORIZONTAL': {
            'kernel': np.array([[-1, -2, -1],
                                [0, 0, 0],
                                [1, 2, 1]]),
        },
        'SOBEL VERTICAL': {
            'kernel': np.array([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]]),
        },
        'PREWITT HORIZONTAL': {
            'kernel': np.array([[-1, -1, -1],
                                [0, 0, 0],
                                [1, 1, 1]]),
        },
        'PREWITT VERTICAL': {
            'kernel': np.array([[-1, 0, 1],
                                [-1, 0, 1],
                                [-1, 0, 1]]),
        }
    }

    # ----------------------- RUNING --------------------------------------
    draw_image(img, title='ORIGINAL IMAGE')

    for title, kernel_data in kernels_dict.items():
        res_img = full_conv(img, kernel_data['kernel'])
        draw_image(res_img, title)


if __name__ == '__main__':
    main()
