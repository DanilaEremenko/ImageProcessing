import cv2
import numpy as np
from my_image_processing import convolve_and_show


def get_gaussian_noise(mean, sigma, shape):
    return np.random.normal(mean, sigma, shape)


def main():
    img = cv2.cvtColor(cv2.imread('images/test_dog.jpg'), cv2.COLOR_BGR2GRAY)

    sigma = int(0.02 * len(np.diag(img)))

    img_noised = img + get_gaussian_noise(mean=0, sigma=sigma, shape=img.shape)

    kernels_dict = {

        'MEAN FILTER': np.array([[1 / 9, 1 / 9, 1 / 9],
                                 [1 / 9, 1 / 9, 1 / 9],
                                 [1 / 9, 1 / 9, 1 / 9]]),

        'GAUSSIAN FILTER': np.array([[1, 4, 7, 4, 1],
                                     [4, 16, 26, 16, 4],
                                     [7, 26, 41, 26, 7],
                                     [4, 16, 26, 16, 4],
                                     [1, 4, 7, 4, 1]]),

    }

    # ----------------------- RUNING --------------------------------------
    convolve_and_show(img, title='ORIGINAL IMAGE')
    convolve_and_show(img_noised, title='NOISED IMAGE')

    for title, kernel in kernels_dict.items():
        convolve_and_show(
            img_arr=img_noised,
            kernel=kernel,
            title=title
        )


if __name__ == '__main__':
    main()
