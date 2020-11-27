import cv2
import numpy as np
from my_image_processing import convolve_and_show


def get_gaussian_noise(mean, sigma, shape):
    return np.random.normal(mean, sigma, shape)


def get_gaussian_kernel(small_img_arr, sigma):
    center_x = small_img_arr.shape[0] // 2
    center_y = small_img_arr.shape[1] // 2
    kernel = np.zeros(small_img_arr.shape)
    for x in range(small_img_arr.shape[0]):
        for y in range(small_img_arr.shape[1]):
            kernel[x][y] = np.exp(-((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * sigma ** 2))

    # after getting kernel
    # w = np.sum(kernel)
    # img_filtered[p_y, p_x, :] = gp / (w + np.finfo(np.float32).eps)

    return kernel


def main():
    img = cv2.cvtColor(cv2.imread('images/test_dog.jpg'), cv2.COLOR_BGR2GRAY)

    sigma = int(0.02 * len(np.diag(img)))

    kernel_len = 5
    # kernel_len = np.int(np.sqrt(sigma) * 3)

    img_noised = img + get_gaussian_noise(mean=0, sigma=sigma, shape=img.shape)

    kernels_dict = {

        'MEAN FILTER': np.array([[1 / 9, 1 / 9, 1 / 9],
                                 [1 / 9, 1 / 9, 1 / 9],
                                 [1 / 9, 1 / 9, 1 / 9]]),

        'GAUSSIAN FILTER': {
            'func': get_gaussian_kernel,
            'shape': (kernel_len, kernel_len),
            'args': {'sigma': sigma}
        },
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
