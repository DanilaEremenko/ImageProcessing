import json

import cv2
import scipy.optimize as optimizers
from convolve_filters import calculate_diff, full_conv, full_adaptive_conv
from noise_filters import get_gaussian_kernel_py, get_bilateral_kernel_np, non_local_means, get_bilateral_kernel_py
import numpy as np
from PIL import Image
import time
import pandas as pd
from lib.plot_part import show_image


######################################################################
# -------------------- NOISE  ----------------------------------------
######################################################################

def get_gaussian_noise(mean, sigma, shape):
    return np.random.normal(mean, sigma, shape)


def add_history_step(history, loss, args_json):
    history['loss'].append(loss)
    history['args'].append(args_json)


####################################################################################
def get_mean_results(kernel_len, orig_img, noised_img):
    res_img = full_conv(
        img_arr=noised_img,
        kernel=np.ones(shape=(kernel_len, kernel_len)) / (kernel_len * kernel_len)
    )
    return res_img, calculate_diff(orig_img, res_img)


def mean_min_func(args, name, orig_img, noised_img, history):
    kernel_len = args[0]
    kernel_len = int(kernel_len)

    res_img, diff = get_mean_results(kernel_len=kernel_len, orig_img=orig_img, noised_img=noised_img)

    print(f"{len(history['loss']) + 1}.{name}: {diff}")

    add_history_step(
        history=history,
        loss=diff,
        args_json=json.dumps({'kernel_len': kernel_len})
    )
    return diff


####################################################################################
def get_gaussian_results(kernel_len, sigma, orig_img, noised_img):
    res_img = full_conv(
        img_arr=noised_img,
        kernel=get_gaussian_kernel_py(kernel_len, sigma)
    )
    return res_img, calculate_diff(orig_img, res_img)


def gaussian_min_func(args, name, orig_img, noised_img, history):
    sigma, kernel_len = args
    kernel_len = int(kernel_len)

    res_img, diff = get_gaussian_results(sigma=sigma, kernel_len=kernel_len, orig_img=orig_img,
                                         noised_img=noised_img)

    print(f"{len(history['loss']) + 1}.{name}: {diff}")

    add_history_step(
        history=history,
        loss=diff,
        args_json=json.dumps({'sigma': sigma, 'kernel_len': kernel_len})
    )
    return diff


####################################################################################
def get_bilateral_results(sigma_i, sigma_s, kernel_len, orig_img, noised_img):
    res_img = full_adaptive_conv(
        img_arr=noised_img,
        kernel_args={'sigma_i': sigma_i, 'sigma_s': sigma_s},
        kernel_data={
            'func': get_bilateral_kernel_py,
            'adaptive_kernel': False,
            'shape': (kernel_len, kernel_len),
        },
    )
    return res_img, calculate_diff(orig_img, res_img)


def bilateral_min_func(args, name, orig_img, noised_img, history):
    sigma_i, sigma_s, kernel_len = args
    kernel_len = int(kernel_len)

    res_img, diff = get_bilateral_results(sigma_i=sigma_i, sigma_s=sigma_s, kernel_len=kernel_len,
                                          orig_img=orig_img, noised_img=noised_img)

    print(f"{len(history['loss']) + 1}.{name}: {diff}")

    add_history_step(
        history=history,
        loss=diff,
        args_json=json.dumps({'sigma_i': sigma_i, 'sigma_s': sigma_s, 'kernel_len': kernel_len})
    )
    return diff


####################################################################################
def get_nlm_results(sigma, h, orig_img, noised_img):
    res_img = non_local_means(
        noisy=noised_img,
        bw_size=30,
        sw_size=5,
        h=h,
        sigma=sigma,
        verbose=False
    )
    return res_img, calculate_diff(orig_img, res_img)


def nlm_min_func(args, name, orig_img, noised_img, history):
    sigma = args[0]
    h = 10
    res_img, diff = get_nlm_results(sigma=sigma, h=h, orig_img=orig_img, noised_img=noised_img)

    print(f"{len(history['loss']) + 1}.{name}: {diff}")

    add_history_step(
        history=history,
        loss=diff,
        args_json=json.dumps({'sigma': sigma, 'h': h})
    )
    return diff


####################################################################################
def get_img_from_name(name, orig_img, noised_img, args):
    if name == 'MEAN FILTER':
        kernel_len = args['kernel_len']
        res_img, _ = get_mean_results(orig_img=orig_img, noised_img=noised_img, kernel_len=kernel_len)

    elif name == 'GAUSSIAN FILTER':
        sigma = args['sigma']
        kernel_len = args['kernel_len']
        res_img, _ = get_gaussian_results(orig_img=orig_img, noised_img=noised_img, kernel_len=kernel_len, sigma=sigma)

    elif name == 'PYTHON BILATERAL FILTER':
        sigma_i = args['sigma_i']
        sigma_s = args['sigma_s']
        kernel_len = args['kernel_len']
        res_img, _ = get_bilateral_results(orig_img=orig_img, noised_img=noised_img, sigma_i=sigma_i, sigma_s=sigma_s,
                                           kernel_len=kernel_len)
    elif name == 'Non-local means':
        h = args['h']
        sigma = args['sigma']
        res_img, _ = get_nlm_results(orig_img=orig_img, noised_img=noised_img, sigma=sigma, h=h)
    else:
        raise Exception('Undefined name')

    return res_img


def get_compare_df(orig_img, noised_img, kernel_bounds, sigma_start, eval_num):
    #######################################################
    # ----------------- SIMPLE FILTERS --------------------
    #######################################################
    kernels_dict = {
        'MEAN FILTER': {
            'func': mean_min_func,
            'args': {'kernel_len': kernel_bounds[0]},
            'extra_args': ('MEAN FILTER', orig_img, noised_img),
            'history': {'name': [], 'args': [], 'loss': []}
        },
        'GAUSSIAN FILTER': {
            'func': gaussian_min_func,
            'args': {'sigma': sigma_start, 'kernel_len': kernel_bounds[0]},
            'extra_args': ('GAUSSIAN FILTER', orig_img, noised_img),
            'history': {'name': [], 'args': [], 'loss': []}
        },
        'PYTHON BILATERAL FILTER': {
            'func': bilateral_min_func,
            'args': {'sigma_i': sigma_start, 'sigma_s': sigma_start, 'kernel_len': kernel_bounds[0]},
            'extra_args': ('PYTHON BILATERAL FILTER', orig_img, noised_img),
            'history': {'name': [], 'loss': [], 'args': []}
        }
    }

    # ----------------------- RUNING --------------------------------------
    for name, hyper_data in kernels_dict.items():
        optimizers.fmin_bfgs(
            f=hyper_data['func'],
            x0=np.array(list(hyper_data['args'].values())),
            args=(*hyper_data['extra_args'], hyper_data['history']),
            maxiter=eval_num
        )
        hyper_data['history']['name'] = [name] * (len(hyper_data['history']['loss']))

    res_df = pd.concat(
        [pd.DataFrame(value['history']) for value in kernels_dict.values()],
        sort=True
    )
    return res_df


def get_nlm_df(orig_img, noised_img, sigma_start, eval_num):
    #######################################################
    # ----------------- NLM -------------------------------
    #######################################################
    hyper_data = {
        'func': nlm_min_func,
        'args': {'sigma': sigma_start},
        'extra_args': ('NLM-FILTER', orig_img, noised_img),
        'history': {'name': [], 'loss': [], 'args': []}
    }
    name = 'Non-local means'
    optimizers.fmin_bfgs(
        f=hyper_data['func'],
        x0=np.array(list(hyper_data['args'].values())),
        args=(*hyper_data['extra_args'], hyper_data['history']),
        maxiter=eval_num
    )
    hyper_data['history']['name'] = [name] * (len(hyper_data['history']['loss']))

    res_df = pd.DataFrame(hyper_data['history'])
    return res_df


def main():
    orig_img = cv2.cvtColor(cv2.imread('../dimages/test_dog.jpg'), cv2.COLOR_BGR2GRAY)
    # orig_img = orig_img[300:450, 300:450]
    # original
    orig_img = Image.fromarray(orig_img)
    orig_img.thumbnail((512, 512))
    orig_img = np.asarray(orig_img)
    show_image(orig_img, 'ORIGINAL IMAGE')

    # noise
    NOISE_SIGMA = 10

    noised_img = orig_img + get_gaussian_noise(mean=0, sigma=NOISE_SIGMA, shape=orig_img.shape)
    show_image(noised_img, 'NOISED IMAGE')

    # run search functions
    res_df = pd.concat([
        pd.DataFrame(
            {
                'name': ['noise'],
                'loss': [calculate_diff(orig_img, noised_img)],
                'args': ['']
            }
        ),
        get_compare_df(
            orig_img=orig_img,
            noised_img=noised_img,
            kernel_bounds=(3, 9),
            sigma_start=5,
            eval_num=None
        ),
        get_nlm_df(
            orig_img=orig_img,
            noised_img=noised_img,
            sigma_start=5,
            eval_num=None
        )
    ],
        sort=True
    )
    return res_df


if __name__ == '__main__':
    res_df = main()
    res_df.to_pickle('noise.pkl')
    # res_df = pd.read_pickle('noise.pkl')
