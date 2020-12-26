import cv2
from convolve_filters import convolve_and_show
from noise_filters import get_gaussian_kernel, get_bilateral_kernel_np, non_local_means
from noise_filters_cython import get_bilateral_kernel_cython
import numpy as np
import time


######################################################################
# -------------------- MAIN ------------------------------------------
######################################################################

def get_gaussian_noise(mean, sigma, shape):
    return np.random.normal(mean, sigma, shape)


def main():
    img = cv2.cvtColor(cv2.imread('images/test_dog.jpg'), cv2.COLOR_BGR2GRAY)

    sigma = int(0.02 * len(np.diag(img)))

    kernel_len = 5
    # kernel_len = np.int(np.sqrt(sigma) * 3)

    img_noised = img + get_gaussian_noise(mean=0, sigma=sigma, shape=img.shape)

    kernels_dict = {
        'MEAN FILTER': np.ones(shape=(kernel_len, kernel_len)) / (kernel_len * kernel_len),

        'GAUSSIAN FILTER': get_gaussian_kernel(kernel_len, sigma),

        'NUMPY BILATERAL FILTER': {
            'func': get_bilateral_kernel_np,
            'shape': (kernel_len, kernel_len),
            'args': {'sigma_i': sigma, 'sigma_s': sigma}
        },

        'CYTHON BILATERAL FILTER': {
            'func': get_bilateral_kernel_cython,
            'shape': (kernel_len, kernel_len),
            'args': {'sigma_i': sigma, 'sigma_s': sigma}
        },
    }

    # ----------------------- RUNING --------------------------------------
    convolve_and_show(img, title='ORIGINAL IMAGE')
    convolve_and_show(img_noised, title='NOISED IMAGE')

    for title, kernel in kernels_dict.items():
        start_time = time.time()
        convolve_and_show(
            img_arr=img_noised,
            kernel=kernel,
            title=title + '\n'
        )
        print(f"{title} time = {time.time() - start_time}")

    import matplotlib.pyplot as plt

    start_time = time.time()
    res = non_local_means(
        noisy=img_noised,
        big_window_size=20,
        small_window_size=6,
        sigma=sigma,
        h=14,
        verbose=False
    )
    title = 'Non-local means'
    print(f"{title} time = {time.time() - start_time}")
    plt.imshow(res, cmap='gray')
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    main()
