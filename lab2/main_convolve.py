import cv2
import numpy as np
from convolve_filters import full_conv
from lib.plot_part import draw_images, show_image


def max_array(arr1, arr2):
    assert arr1.shape == arr2.shape
    return np.array(
        [max(el1, el2) for el1, el2 in zip(arr1.flatten(), arr2.flatten())],
        dtype='uint8').reshape(arr1.shape)


def main(img_path, save_path=None):
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    show_image(img_arr=img, title='SRC IMAGE')

    kernels_dict = {

        'ROBERTS LEFT': {
            'kernel': np.array([[1, 0],
                                [0, -1]]),
        },
        'ROBERTS RIGHT': {
            'kernel': np.array([[0, 1],
                                [-1, 0]]),
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
    }

    # ----------------------- RUNING --------------------------------------
    res_dict = {}
    for i, (title, kernel_data) in enumerate(kernels_dict.items()):
        res = full_conv(img, kernel_data['kernel'])
        _, res_dict[title] = cv2.threshold(res, 128, 255, cv2.THRESH_BINARY)
        if i % 2 == 1:
            res_title = list(kernels_dict.keys())[i].split(' ')[0] + " RESULT"
            res_img = max_array(*list(res_dict.values())[-2:])
            res_dict[res_title] = res_img

    draw_images(
        imgs=list(res_dict.values()),
        titles=list(res_dict.keys()),
        plt_shape=(round(len(res_dict) / 3), 3),
        show=True,
        save_path=save_path,
    )


if __name__ == '__main__':
    main(img_path='../dimages/boundaries_yum.jpg', save_path='bound_boundaries_yum.jpg')
    main(img_path='../dimages/test_dog.jpg', save_path='bound_test_dog.jpg')
