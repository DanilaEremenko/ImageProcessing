import cv2

import lib_trim
import lib_equal
import numpy as np
from lib.plot_part import draw_images_and_hists


def build_histogram(img_arr, width):
    result_hist = [0] * width
    for i in img_arr:
        for j in i:
            result_hist[j] += 1
    return result_hist


def main(img_path, save_path, trim_part):
    # src data loading
    src_image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    src_hist = build_histogram(img_arr=src_image, width=256)

    #########################################################################
    # ------------------------- TRIM PART -----------------------------------
    #########################################################################
    left, right, trim_hist = lib_trim.trim_part(trim_part, src_hist)

    # transform
    transform_trim_matrix = lib_trim.build_change_matrix(
        new_left=0,
        new_right=256,
        left=left,
        right=right
    )
    res_trim_image = np.array(
        [transform_trim_matrix[pix] for pix in src_image.flatten()],
        dtype='uint8'
    ).reshape(src_image.shape)

    # final hist building
    final_trim_hist = build_histogram(img_arr=res_trim_image, width=256)

    ###########################################################################
    # -------------------------- EQUAL PART -----------------------------------
    ###########################################################################
    base_eq_cumul_hist = lib_equal.build_cumulative(src_hist)

    # transform
    transform_eq_matrix = lib_equal.build_equalize_matrix(base_eq_cumul_hist)

    res_eq_image = np.array(
        [transform_eq_matrix[pix] for pix in src_image.flatten()],
        dtype='uint8'
    ).reshape(src_image.shape)

    # final hist building
    final_eq_hist = build_histogram(img_arr=res_eq_image, width=256)

    ###########################################################################
    # -------------------------- DRAW RESULTS ---------------------------------
    ###########################################################################
    draw_images_and_hists(
        imgs=[src_image, res_trim_image, res_eq_image],
        hists=[src_hist, final_trim_hist, final_eq_hist],
        titles=['src trim', 'final trim', 'final eq'],
        save_path=save_path
    )


if __name__ == '__main__':
    main(img_path="../dimages/test_dog.jpg", save_path='contrast_test_dog.png', trim_part=0.10)
    main(img_path="../dimages/new_york.webp", save_path='contrast_new_york.png', trim_part=0.10)
    main(img_path="../dimages/bad_brightness.jpeg", save_path='contrast_bad_brightness.png', trim_part=0.10)
