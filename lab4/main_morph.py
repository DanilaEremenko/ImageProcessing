import cv2
import numpy as np
from lib.plot_part import draw_images


# делитация
def dilation(A, B):
    B_h, B_w = B.shape
    A_h, A_w = A.shape
    res_dilation = A.copy()
    target_row, target_col = np.where(A == 255)
    # для каждого пикселя ищем пересечение
    for ai, aj in zip(target_row, target_col):
        for bi in range(B_h):
            for bj in range(B_w):
                checking_row = ai + bi - B_h // 2
                checking_col = aj + bj - B_w // 2
                if 0 <= checking_row <= A_h - 1 and 0 <= checking_col <= A_w - 1:
                    # якорная точка
                    res_dilation[checking_row][checking_col] = B[bi][bj]
    return res_dilation


# эрозия
def erosion(A, B):
    B_h, B_w = B.shape
    A_h, A_w = A.shape
    res_erosion = A.copy()
    target_row, target_col = np.where(A == 255)
    # ищем все смещения, где B полностью входит в A
    for ai, aj in zip(target_row, target_col):
        for bi in range(B_h):
            for bj in range(B_w):
                checking_row = ai + bi - B_h // 2
                checking_col = aj + bj - B_w // 2
                if 0 <= checking_row <= A_h - 1 and 0 <= checking_col <= A_w - 1:
                    # если B не является подмножеством A, то точка равна 0
                    if A[checking_row][checking_col] != B[bi][bj]:
                        res_erosion[ai][aj] = 0
                else:
                    res_erosion[ai][aj] = 0
    return res_erosion


# открытие
def opening(A, B):
    return dilation(erosion(A, B), B)


# закрытие
def closing(A, B):
    return erosion(dilation(A, B), B)


def main(img_path):
    src_img = cv2.imread(img_path, 0)

    # бинаризация
    _, binary = cv2.threshold(src_img, 128, 255, cv2.THRESH_BINARY)
    A = np.array([255 if el == 0 else 0 for el in binary.flatten()]).reshape(src_img.shape)

    window_shape = 9
    B = np.ones(shape=(window_shape, window_shape), dtype=np.uint8) * 255

    img_dilation = dilation(A, B)

    img_erosion = erosion(A, B)

    img_opening = opening(A, B)

    img_closing = closing(A, B)

    draw_images(
        imgs=[A, img_dilation, img_erosion, img_opening, img_closing],
        titles=['src', 'dilation', 'erosion', 'opening', 'closing'],
        show=True
    )


if __name__ == '__main__':
    main(img_path="../dimages/boundaries_yum.jpg")
