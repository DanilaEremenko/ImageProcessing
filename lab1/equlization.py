import cv2
from plot_part import draw_image, draw_hist


def build_histogram(values, width):
    result_hist = [0] * width
    for i in values:
        for j in i:
            result_hist[j] += 1
    return result_hist


def build_cumulative(hist):
    result_array = [0] * len(hist)
    result_array[0] = hist[0]
    for i in range(1, len(hist)):
        result_array[i] = result_array[i - 1] + hist[i]
    return [number / sum(hist) for number in result_array]


def normalize(array):
    lbound = min(array)
    rbound = max(array)
    width = rbound - lbound
    scale = 255 / width
    normalized_array = [(x - lbound) * scale for x in array]
    return normalized_array


def build_equalize_matrix(norm_cumulative):
    result_matrix = [n_el * len(norm_cumulative) for n_el in norm_cumulative]
    return normalize(result_matrix)


def main():
    image = cv2.imread("../dimages/test_dog.jpg", 0)
    draw_image(image, 'input image')

    base_hist = build_histogram(image, 256)
    draw_hist(base_hist, 'base histogram')

    base_cumul = build_cumulative(base_hist)
    draw_hist(base_cumul, 'base cumulative')

    transform_matrix = build_equalize_matrix(base_cumul)
    draw_hist(transform_matrix, 'transform matrix')

    res_image = image.copy()
    for i in range(len(image)):
        for j in range(len(image[0])):
            res_image[i][j] = transform_matrix[image[i][j]]

    final_hist = build_histogram(image, 256)
    draw_hist(final_hist, 'final histogram')

    final_cumul = build_cumulative(final_hist)
    draw_hist(final_cumul, 'final cumulative')

    draw_image(image, 'result image')


if __name__ == '__main__':
    main()
