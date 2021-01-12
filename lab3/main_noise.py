import json

import cv2
from convolve_filters import draw_image, calculate_diff, full_conv, full_adaptive_conv
from noise_filters import get_gaussian_kernel_py, get_bilateral_kernel_np, non_local_means, get_bilateral_kernel_py
from noise_filters_cython import get_bilateral_kernel_cython
import numpy as np
import time
import pandas as pd


######################################################################
# -------------------- MAIN ------------------------------------------
######################################################################

def get_gaussian_noise(mean, sigma, shape):
    return np.random.normal(mean, sigma, shape)


def compare_filters():
    orig_img = cv2.cvtColor(cv2.imread('../dimages/test_dog.jpg'), cv2.COLOR_BGR2GRAY)
    from PIL import Image
    orig_img = Image.fromarray(orig_img)
    orig_img.thumbnail((512, 512))
    orig_img = np.asarray(orig_img)
    draw_image(orig_img, 'ORIGINAL IMAGE')

    NOISE_SIGMA = 10
    KERNEL_LEN = 5
    SIGMA_VALUES = np.linspace(2, 20, 5)

    img_noised = orig_img + get_gaussian_noise(mean=0, sigma=NOISE_SIGMA, shape=orig_img.shape)
    draw_image(img_noised, 'NOISED IMAGE')

    #######################################################
    # ----------------- SIMPLE FILTERS --------------------
    #######################################################
    kernels_dict = {
        'MEAN FILTER': {
            'kernel': np.ones(shape=(KERNEL_LEN, KERNEL_LEN)) / (KERNEL_LEN * KERNEL_LEN),
            'adaptive_kernel': False,
            'args_list': [{}]
        },
        'GAUSSIAN FILTER': {
            'func': get_gaussian_kernel_py,
            'adaptive_kernel': False,
            'shape': (KERNEL_LEN, KERNEL_LEN),
            'args_list': [{'sigma': curr_sigma} for curr_sigma in SIGMA_VALUES],
        },
        'PYTHON BILATERAL FILTER': {
            'func': get_bilateral_kernel_py,
            'adaptive_kernel': True,
            'shape': (KERNEL_LEN, KERNEL_LEN),
            'args_list': [{'sigma_i': curr_sigma, 'sigma_s': NOISE_SIGMA / 2} for curr_sigma in SIGMA_VALUES],
        }
    }

    # ----------------------- RUNING --------------------------------------
    res_df = pd.DataFrame({'name': [], 'diff': [], 'time': [], 'args': [], 'res_img': []})

    for name, kernel_data in kernels_dict.items():
        for args in kernel_data['args_list']:
            start_time = time.time()
            if kernel_data['adaptive_kernel']:
                res_img = full_adaptive_conv(img_noised, kernel_data, args)
            elif 'kernel' in kernel_data.keys():
                res_img = full_conv(img_noised, kernel_data['kernel'])
            else:
                res_img = full_conv(img_noised, kernel_data['func'](len(kernel_data['shape']), **args))

            curr_time = time.time() - start_time
            diff = calculate_diff(orig_img, res_img)

            res_df = res_df.append(
                pd.DataFrame({'name': [name], 'diff': [diff], 'time': [curr_time],
                              'args': json.dumps(args),
                              'res_img': [res_img]})
            )
            print(f"{name} with {args} done in {round(curr_time, 2)}s")
    #######################################################
    # ----------------- NLM -------------------------------
    #######################################################
    name = 'Non-local means'
    for curr_sigma in SIGMA_VALUES:
        start_time = time.time()
        res_img = non_local_means(
            noisy=img_noised,
            bw_size=17,
            sw_size=7,
            sigma=curr_sigma,
            verbose=False
        )
        curr_time = time.time() - start_time
        diff = calculate_diff(orig_img, res_img)

        res_df = res_df.append(
            pd.DataFrame({'name': [name], 'diff': [diff], 'time': [curr_time],
                          'args': json.dumps({'sigma': curr_sigma}),
                          'res_img': [res_img]})
        )
        print(f"{name} with sigma = {curr_sigma} done in {round(curr_time, 2)}s")

    return res_df


if __name__ == '__main__':
    res_df = compare_filters()
