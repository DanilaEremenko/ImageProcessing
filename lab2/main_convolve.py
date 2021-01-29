import cv2
import numpy as np
from convolve_filters import full_conv
from lib.plot_part import draw_images


def main(img_path):
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)

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
    res_dict = {'src': img}
    for title, kernel_data in kernels_dict.items():
        res = full_conv(img, kernel_data['kernel'])
        _, res_dict[title] = cv2.threshold(res, 128, 255, cv2.THRESH_BINARY)

    draw_images(
        imgs=list(res_dict.values()),
        titles=list(res_dict.keys()),
        show=True
    )


if __name__ == '__main__':
    main(img_path='../dimages/boundaries_yum.jpg')
    main(img_path='../dimages/test_dog.jpg')
