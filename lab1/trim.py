import cv2
import matplotlib.pyplot as plt
import math


def build_histogram(values, width, left, right):
    result_hist = [0] * width
    for line in values:
        for element in line:
            if element < left:
                continue
            if element > right:
                continue
            result_hist[element] += 1
    return result_hist


def trim_percent(percent, original_hist):
    left = 0
    right = 255
    total_space = sum(original_hist)
    new_space = total_space
    print(new_space / total_space)
    while True:
        left += 1
        new_space = sum(original_hist[left:right])
        print(new_space / total_space)
        print(str(left) + " - " + str(right))
        if new_space / total_space < 1.0 - percent:
            return [left, right]
        right -= 1
        new_space = sum(original_hist[left:right])
        print(new_space / total_space)
        print(str(left) + " - " + str(right))
        if new_space / total_space < 1.0 - percent:
            return [left, right]


def trim_percent2(percent, original_hist):
    left = 0
    right = 255
    leftCut = sum(original_hist) * percent // 2
    rightCut = leftCut
    print(str(leftCut) + " - value to cut off from " + str(sum(original_hist)))
    while leftCut > 0:
        if original_hist[left] < leftCut:
            leftCut -= original_hist[left]
            original_hist[left] = 0
            left += 1
        else:
            original_hist[left] -= leftCut
            leftCut = 0
    while rightCut > 0:
        if original_hist[right] < rightCut:
            rightCut -= original_hist[right]
            original_hist[right] = 0
            right -= 1
        else:
            original_hist[right] -= rightCut
            rightCut = 0
    print("new borders: " + str(left) + " - " + str(right))
    return [left, right, original_hist]


def build_change_matrix(width, a, b):
    c = 0
    d = width
    result_matrix = [0] * width
    # (i - a) * ((d - c) / (b - a)) + c
    new_range = d - c
    old_range = b - a
    range_multiplier = new_range / old_range
    for i in range(len(result_matrix)):
        new_color = ((i - a) * range_multiplier + c) // 1
        result_matrix[i] = max(min(new_color, 255), 0)
    return result_matrix


def main():
    originalImage = cv2.imread("images/test_dog.jpg")
    image = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Original", image)

    baseHistogram = build_histogram(image, 256, 0, 255)
    base2 = baseHistogram
    plt.plot(baseHistogram)
    plt.show()

    test = trim_percent2(0.07, base2)
    print(test[0:2])
    trimmedTest = test[2]
    plt.plot(trimmedTest)
    plt.show()

    transform_matrix = build_change_matrix(256, test[0], test[1])
    # imageCopy = image
    for i in range(len(image)):
        for j in range(len(image[0])):
            image[i][j] = transform_matrix[image[i][j]]

    finalHistogram = build_histogram(image, 256, 0, 255)
    plt.plot(finalHistogram)
    plt.show()

    cv2.imshow("Result", image)

    cv2.waitKey(0)


if __name__ == '__main__':
    main()
