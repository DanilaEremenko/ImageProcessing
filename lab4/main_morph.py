import cv2
import numpy as np
from matplotlib import pyplot as plt


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


def draw_image(img_arr, title):
    plt.imshow(img_arr, cmap='gray')
    plt.title(title)
    plt.show()


def draw_hist(hist, title):
    plt.plot(hist)
    plt.title(title)
    plt.show()


def main():
    img = cv2.imread("../dimages/boundaries_yum.jpg", 0)
    draw_image(img, 'Source image')

    # бинаризация
    _, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    A = np.array([255 if el == 0 else 0 for el in binary.flatten()]).reshape(img.shape)
    draw_image(A, 'Binary inverted image')

    window_shape = 9
    B = np.ones(shape=(window_shape, window_shape), dtype=np.uint8) * 255

    img_dilation = dilation(A, B)
    draw_image(img_dilation, 'Dilation')

    img_erosion = erosion(A, B)
    draw_image(img_erosion, 'Erosion')

    img_opening = opening(A, B)
    draw_image(img_opening, 'Opening')

    img_closing = closing(A, B)
    draw_image(img_closing, 'Closing')


if __name__ == '__main__':
    main()
