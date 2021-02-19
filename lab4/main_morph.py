import cv2
import numpy as np
from lib.plot_part import draw_images, show_image


def dilation(A, B, strong_pix=255):
    B_h, B_w = B.shape
    A_h, A_w = A.shape
    res_dilation = A.copy()
    strong_ids = np.where(A == strong_pix)
    # A = B
    for ai, aj in zip(*strong_ids):
        for bi in range(B_h):
            for bj in range(B_w):
                neigh_row = ai + bi - B_h // 2
                neigh_col = aj + bj - B_w // 2
                if 0 <= neigh_row < A_h and 0 <= neigh_col < A_w:
                    # якорная точка
                    res_dilation[neigh_row][neigh_col] = B[bi][bj]
    return res_dilation


def erosion(A, B, strong_pix=255, weak_pix=0):
    B_h, B_w = B.shape
    A_h, A_w = A.shape
    res_erosion = A.copy()
    strong_ids = np.where(A == strong_pix)
    # found where B in A
    for ai, aj in zip(*strong_ids):
        for bi in range(B_h):
            for bj in range(B_w):
                neigh_row = ai + bi - B_h // 2
                neigh_col = aj + bj - B_w // 2
                if 0 <= neigh_row < A_h and 0 <= neigh_col < A_w:
                    # curr_B not in A -> pix = weak_pix
                    if A[neigh_row][neigh_col] != B[bi][bj]:
                        res_erosion[ai][aj] = weak_pix
                        break
                else:
                    res_erosion[ai][aj] = weak_pix
    return res_erosion


def opening(A, B):
    return dilation(erosion(A, B), B)


def closing(A, B):
    return erosion(dilation(A, B), B)


def main(img_path, window_shape, invert=True, save_path=None):
    src_img = cv2.imread(img_path, 0)

    # бинаризация
    _, binary = cv2.threshold(src_img, 128, 255, cv2.THRESH_BINARY)
    if invert:
        A = np.array([255 if el == 0 else 0 for el in binary.flatten()]).reshape(src_img.shape)
    else:
        A = binary
    show_image(A, 'SRC IMAGE')

    B = np.ones(shape=(window_shape, window_shape), dtype=np.uint8) * 255

    img_dilation = dilation(A, B)

    img_erosion = erosion(A, B)

    img_opening = opening(A, B)

    img_closing = closing(A, B)

    draw_images(
        imgs=[img_dilation, img_erosion, img_opening, img_closing],
        titles=['DILATION', 'EROSION', 'OPENING', 'CLOSING'],
        plt_shape=(2, 2),
        show=True,
        save_path=save_path
    )


if __name__ == '__main__':
    main(img_path="../dimages/boundaries_yum.jpg", invert=True, window_shape=9, save_path='morph_boundaries_yum.jpg')
    main(img_path="../dimages/j_noised.PNG", invert=False, window_shape=3, save_path='j_noised.PNG')
    main(img_path="../dimages/j_leaky.PNG", invert=False, window_shape=3, save_path='j_leaky.PNG')
