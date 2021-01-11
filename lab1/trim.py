import cv2
import matplotlib.pyplot as plt


def draw_image(img_arr, title):
    plt.imshow(img_arr, cmap='gray')
    plt.title(title)
    plt.show()


def draw_hist(hist, title):
    plt.plot(hist)
    plt.title(title)
    plt.show()


def build_histogram(img_arr, width, left, right):
    result_hist = [0] * width
    for el in img_arr.flatten():
        if left < el < right:
            result_hist[el] += 1
    return result_hist


def build_change_matrix(width, left, right):
    a = left
    b = right
    c = 0
    d = width
    result_matrix = [0] * width
    # (i - a) * ((d - c) / (b - a)) + c
    new_range = d - c
    old_range = b - a
    range_multiplier = new_range / old_range
    for i in range(len(result_matrix)):
        new_color = int((i - a) * range_multiplier + c)
        result_matrix[i] = max(min(new_color, 255), 0)
    return result_matrix


def trim_part(part, hist, left=0, right=255):
    res_hist = hist.copy()

    left_cut = sum(res_hist) * part // 2
    right_cut = left_cut

    print(f"{left_cut} - value to cut off from {sum(res_hist)}")

    while left_cut > 0:
        if hist[left] < left_cut:
            left_cut -= res_hist[left]
            res_hist[left] = 0
            left += 1
        else:
            res_hist[left] -= left_cut
            left_cut = 0
    while right_cut > 0:
        if res_hist[right] < right_cut:
            right_cut -= res_hist[right]
            res_hist[right] = 0
            right -= 1
        else:
            res_hist[right] -= right_cut
            right_cut = 0
    print("new borders: " + str(left) + " - " + str(right))
    return left, right, res_hist


def main():
    image = cv2.cvtColor(cv2.imread("../dimages/test_dog.jpg"), cv2.COLOR_BGR2GRAY)

    # show input
    draw_image(image, 'source')
    base_hist = build_histogram(
        img_arr=image,
        width=256,
        left=0,
        right=255
    )
    draw_hist(base_hist, 'base histogram')

    # trim
    left, right, trim_hist = trim_part(0.07, base_hist)
    draw_hist(trim_hist, 'trim histogram')

    # transform
    transform_matrix = build_change_matrix(
        width=256,
        left=left,
        right=right
    )
    draw_hist(transform_matrix, 'transform matrix')
    res_image = image.copy()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            res_image[i][j] = transform_matrix[image[i][j]]

    # show results
    draw_image(res_image, 'result')
    final_hist = build_histogram(
        img_arr=res_image,
        width=256,
        left=0,
        right=255
    )
    draw_hist(final_hist, 'final histogram')


if __name__ == '__main__':
    main()
