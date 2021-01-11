import cv2
from convolve_filters import convolve_and_show
from noise_filters import get_gaussian_kernel_py, get_bilateral_kernel_np, non_local_means, get_bilateral_kernel_py
from noise_filters_cython import get_bilateral_kernel_cython
import numpy as np
import time


######################################################################
# -------------------- MAIN ------------------------------------------
######################################################################

def get_gaussian_noise(mean, sigma, shape):
    return np.random.normal(mean, sigma, shape)


def main():
    img = cv2.cvtColor(cv2.imread('../dimages/test_dog.jpg'), cv2.COLOR_BGR2GRAY)
    from PIL import Image
    img = Image.fromarray(img)
    img.thumbnail((512, 512))
    img = np.asarray(img)

    sigma = int(0.02 * len(np.diag(img)))

    kernel_len = 5
    # kernel_len = np.int(np.sqrt(sigma) * 3)

    img_noised = img + get_gaussian_noise(mean=0, sigma=sigma, shape=img.shape)

    kernels_dict = {
        'MEAN FILTER': np.ones(shape=(kernel_len, kernel_len)) / (kernel_len * kernel_len),

        'GAUSSIAN FILTER': get_gaussian_kernel_py(kernel_len, sigma),

        'NUMPY BILATERAL FILTER': {
            'func': get_bilateral_kernel_np,
            'shape': (kernel_len, kernel_len),
            'args': {'sigma_i': sigma * 3, 'sigma_s': sigma / 2}
        },
        'PYTHON BILATERAL FILTER': {
            'func': get_bilateral_kernel_py,
            'shape': (kernel_len, kernel_len),
            'args': {'sigma_i': sigma * 3, 'sigma_s': sigma / 2}
        },

        'CYTHON BILATERAL FILTER': {
            'func': get_bilateral_kernel_cython,
            'shape': (kernel_len, kernel_len),
            'args': {'sigma_i': sigma * 3, 'sigma_s': sigma / 2}
        },
    }

    # ----------------------- RUNING --------------------------------------
    convolve_and_show(img, title='ORIGINAL IMAGE', original_image=img)
    convolve_and_show(img_noised, title='NOISED IMAGE', original_image=img)

    for title, kernel in kernels_dict.items():
        start_time = time.time()
        convolve_and_show(
            img_arr=img_noised,
            kernel=kernel,
            title=title,
            original_image=img
        )
        print(f"{title} time = {time.time() - start_time}")

    start_time = time.time()
    res = non_local_means(
        noisy=img_noised,
        bw_size=20,
        sw_size=6,
        sigma=sigma,
        h=67,
        verbose=True
    )
    title = 'Non-local means'
    print(f"{title} time = {time.time() - start_time}")
    convolve_and_show(img_arr=res, title=title, original_image=img)


if __name__ == '__main__':
    main()
