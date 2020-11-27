import cv2
import numpy as np
from my_image_processing import convolve_and_show


def main():
    img = cv2.cvtColor(cv2.imread('images/boundaries_yum.jpg'), cv2.COLOR_BGR2GRAY)

    kernels_dict = {

        'ROBERTS LEFT': np.array([[1, 0],
                                  [0, -1]]),

        'ROBERTS RIGHT': np.array([[0, 1],
                                   [-1, 0]]),

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
    convolve_and_show(img, title='ORIGINAL IMAGE')

    for title, kernel in kernels_dict.items():
        convolve_and_show(
            img_arr=img,
            kernel=kernel,
            title=title
        )


if __name__ == '__main__':
    main()
