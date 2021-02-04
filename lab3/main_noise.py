import json

import cv2
import scipy.optimize as optimizers
from lab2.convolve_filters import calculate_diff, full_conv, full_adaptive_conv
from noise_filters import get_gaussian_kernel_py, get_bilateral_kernel_np, non_local_means, get_bilateral_kernel_py
import numpy as np
from PIL import Image
import time
import pandas as pd
from lib.plot_part import show_image, draw_image_and_loss, draw_images


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
def get_nlm_results(sigma, bw_size, sw_size, h_weight, orig_img, noised_img):
    res_img = non_local_means(
        noisy=noised_img,
        bw_size=bw_size,
        sw_size=sw_size,
        h_weight=h_weight,
        sigma=sigma,
        verbose=False
    )
    return res_img, calculate_diff(orig_img, res_img)


def nlm_min_func(args, name, orig_img, noised_img, bw_size, sw_size, h_weight, history):
    sigma = args[0]

    res_img, diff = get_nlm_results(
        sigma=sigma,
        bw_size=bw_size,
        sw_size=sw_size,
        h_weight=h_weight,
        orig_img=orig_img,
        noised_img=noised_img
    )

    print(f"{len(history['loss']) + 1}.{name}: {diff}")

    add_history_step(
        history=history,
        loss=diff,
        args_json=json.dumps({'sigma': sigma, 'h_weight': h_weight, 'bw_size': bw_size, 'sw_size': sw_size})
    )
    return diff


####################################################################################
def get_res_from_name(name, orig_img, noised_img, args):
    if name == 'MEAN FILTER':
        kernel_len = args['kernel_len']
        res_img, loss = get_mean_results(
            orig_img=orig_img, noised_img=noised_img,
            kernel_len=kernel_len
        )

    elif name == 'GAUSSIAN FILTER':
        sigma = args['sigma']
        kernel_len = args['kernel_len']
        res_img, loss = get_gaussian_results(
            orig_img=orig_img, noised_img=noised_img,
            kernel_len=kernel_len, sigma=sigma
        )

    elif name == 'PYTHON BILATERAL FILTER':
        sigma_i = args['sigma_i']
        sigma_s = args['sigma_s']
        kernel_len = args['kernel_len']
        res_img, loss = get_bilateral_results(
            orig_img=orig_img, noised_img=noised_img,
            sigma_i=sigma_i, sigma_s=sigma_s, kernel_len=kernel_len
        )
    elif name == 'NLM':
        bw_size = args['bw_size']
        sw_size = args['sw_size']
        h_weight = args['h_weight']
        sigma = args['sigma']
        res_img, loss = get_nlm_results(
            orig_img=orig_img, noised_img=noised_img,
            sigma=sigma, bw_size=bw_size, sw_size=sw_size, h_weight=h_weight
        )
    else:
        raise Exception('Undefined name')

    return res_img, loss


def get_compare_df(orig_img, noised_img, kernel_len_start, sigma_start, eval_num):
    #######################################################
    # ----------------- SIMPLE FILTERS --------------------
    #######################################################
    kernels_dict = {
        'MEAN FILTER': {
            'func': mean_min_func,
            'args': {'kernel_len': kernel_len_start},
            'extra_args': ('MEAN FILTER', orig_img, noised_img),
            'history': {'name': [], 'args': [], 'loss': []}
        },
        'GAUSSIAN FILTER': {
            'func': gaussian_min_func,
            'args': {'sigma': sigma_start, 'kernel_len': kernel_len_start},
            'extra_args': ('GAUSSIAN FILTER', orig_img, noised_img),
            'history': {'name': [], 'args': [], 'loss': []}
        },
        'PYTHON BILATERAL FILTER': {
            'func': bilateral_min_func,
            'args': {'sigma_i': sigma_start, 'sigma_s': sigma_start, 'kernel_len': kernel_len_start},
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


def get_nlm_df(orig_img, noised_img, sigma_start, eval_num, bw_size, sw_size, h_weight):
    #######################################################
    # ----------------- NLM -------------------------------
    #######################################################
    hyper_data = {
        'func': nlm_min_func,
        'args': {'sigma': sigma_start},
        'extra_args': ('NLM-FILTER', orig_img, noised_img, bw_size, sw_size, h_weight),
        'history': {'name': [], 'loss': [], 'args': []}
    }
    name = 'NLM'
    optimizers.fmin_bfgs(
        f=hyper_data['func'],
        x0=np.array(list(hyper_data['args'].values())),
        args=(*hyper_data['extra_args'], hyper_data['history']),
        maxiter=eval_num
    )
    hyper_data['history']['name'] = [name] * (len(hyper_data['history']['loss']))

    res_df = pd.DataFrame(hyper_data['history'])
    return res_df


def get_original_and_noised(img_path, noise_sigma):
    orig_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    # orig_img = orig_img[300:450, 300:450]

    # original
    orig_img = Image.fromarray(orig_img)
    orig_img.thumbnail((512, 512))
    orig_img = np.asarray(orig_img)
    # noised
    noised_img = orig_img + get_gaussian_noise(mean=0, sigma=noise_sigma, shape=orig_img.shape)
    return orig_img, noised_img


def write_res(img_path, save_path, noise_sigma):
    orig_img, noised_img = get_original_and_noised(img_path, noise_sigma)

    show_image(orig_img, 'ORIGINAL IMAGE')
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
            kernel_len_start=3,
            sigma_start=noise_sigma / 2,
            eval_num=5
        ),
        get_nlm_df(
            orig_img=orig_img,
            noised_img=noised_img,
            sigma_start=noise_sigma / 2,
            bw_size=15,
            sw_size=7,
            h_weight=0.5,
            eval_num=5
        )
    ],
        sort=True
    )
    res_df.to_pickle(save_path)
    return res_df


def draw_res(image_path, noise_sigma):
    img_name = image_path.split('/')[-1].split('.')[0]
    res_name = f'{img_name}_{noise_sigma}'
    res_path = f'noise_{res_name}.pkl'

    # loading curr_df and selecting columns
    curr_df = pd.read_pickle(res_path)
    loss_dict = {
        name.upper(): {'loss_history': list(curr_df[curr_df['name'] == name]['loss'])}
        for name in set(curr_df['name']) if name != 'noise'
    }

    # ordering methods
    loss_dict = {
        'MEAN FILTER': loss_dict['MEAN FILTER'],
        'GAUSSIAN FILTER': loss_dict['GAUSSIAN FILTER'],
        'PYTHON BILATERAL FILTER': loss_dict['PYTHON BILATERAL FILTER'],
        'NLM': loss_dict['NLM']
    }
    # parse original and noised image
    orig_img, noised_img = get_original_and_noised(image_path, noise_sigma)

    # getting best results
    for name in loss_dict.keys():
        curr_res = curr_df[curr_df['name'] == name]
        curr_best = curr_res[curr_res['loss'] == curr_res['loss'].min()].iloc[0]

        res_img, actual_loss = get_res_from_name(
            name=name,
            orig_img=orig_img,
            noised_img=noised_img,
            args=json.loads(str(curr_best['args']))
        )
        # assert curr_best['loss'] == actual_loss
        loss_dict[name]['img'] = res_img
        loss_dict[name]['best_loss'] = actual_loss

    draw_image_and_loss(
        loss_keys=list(loss_dict.keys()),
        loss_list=[res_dict['loss_history'] for res_dict in loss_dict.values()],
        title=f'{img_name} with sigma = {noise_sigma}',
        show=True,
        save_path=f'noise_history_{img_name}_with_sigma_{noise_sigma}.png'
    )

    draw_images(
        imgs=[orig_img, noised_img, *[res_dict['img'] for res_dict in loss_dict.values()]],
        titles=['orig', f'noised with sigma = {noise_sigma}',
                *[f"{name} with loss = %.2f" % res_dict['best_loss'] for name, res_dict in loss_dict.items()]
                ],
        plt_shape=(3, 2),
        show=True,
        save_path=f'noise_images_{img_name}_with_sigma_{noise_sigma}.png'
    )


if __name__ == '__main__':
    draw_results = True
    for image_path in ('../dimages/new_york.webp', '../dimages/test_dog.jpg'):
        for noise_sigma in (10, 15):
            if draw_results:
                draw_res(image_path=image_path, noise_sigma=noise_sigma)
            else:
                img_name = image_path.split('/')[-1].split('.')[0]
                write_res(img_path=image_path, save_path=f'noise_{img_name}_{noise_sigma}.pkl', noise_sigma=noise_sigma)
