import numpy as np
import os
import cv2

from lab3.noise_filters import get_gaussian_kernel
from lab2.convolve_filters import full_conv


class CannyEdgeDetector:
    def __init__(self, sigma=1.0, kernel_size=5,
                 weak_pixel=75, strong_pixel=255,
                 lowthreshold=0.05, highthreshold=0.15):
        self.img_smoothed = None
        self.grad_intens_matrix = None
        self.grad_direct_matrix = None
        self.non_max_img = None
        self.threshold_img = None
        self.weak_pixel = weak_pixel
        self.strong_pixel = strong_pixel
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.low_threshold = lowthreshold
        self.high_threshold = highthreshold
        return

    def sobel_filters(self, img):
        sobel_vertical = np.array([[-1, 0, 1],
                                   [-2, 0, 2],
                                   [-1, 0, 1]], np.float32)
        sobel_horizontal = np.array([[1, 2, 1],
                                     [0, 0, 0],
                                     [-1, -2, -1]], np.float32)

        conv_vertical = full_conv(img_arr=img, kernel_data=sobel_vertical)
        conv_horizontal = full_conv(img_arr=img, kernel_data=sobel_horizontal)
        grad_matrix = np.sqrt(np.square(conv_vertical) + np.square(conv_horizontal))
        grad_matrix = grad_matrix / grad_matrix.max() * 255
        theta = np.arctan2(conv_horizontal, conv_vertical)
        return grad_matrix, theta

    def non_max_suppression(self, intensity_matrix, direction_matrix):
        """
        make same thickness and intensity for boundaries
        """
        thinned_img = np.zeros(intensity_matrix.shape, dtype=np.int32)
        angle_matrix = direction_matrix * 180. / np.pi
        angle_matrix[angle_matrix < 0] += 180

        for y in range(1, intensity_matrix.shape[0] - 1):
            for x in range(1, intensity_matrix.shape[1] - 1):
                q = 255
                r = 255

                # angle 0
                if (0 <= angle_matrix[y, x] < 22.5) or (157.5 <= angle_matrix[y, x] <= 180):
                    q = intensity_matrix[y, x + 1]
                    r = intensity_matrix[y, x - 1]
                # angle 45
                elif (22.5 <= angle_matrix[y, x] < 67.5):
                    q = intensity_matrix[y + 1, x - 1]
                    r = intensity_matrix[y - 1, x + 1]
                # angle 90
                elif (67.5 <= angle_matrix[y, x] < 112.5):
                    q = intensity_matrix[y + 1, x]
                    r = intensity_matrix[y - 1, x]
                # angle 135
                elif (112.5 <= angle_matrix[y, x] < 157.5):
                    q = intensity_matrix[y - 1, x - 1]
                    r = intensity_matrix[y + 1, x + 1]

                if (intensity_matrix[y, x] >= q) and (intensity_matrix[y, x] >= r):
                    thinned_img[y, x] = intensity_matrix[y, x]
                else:
                    thinned_img[y, x] = 0

        return thinned_img

    def threshold(self, img):

        high_threshold = img.max() * self.high_threshold
        low_threshold = high_threshold * self.low_threshold

        res = np.zeros(img.shape, dtype=np.int32)

        strong_i, strong_j = np.where(img >= high_threshold)
        weak_i, weak_j = np.where((img <= high_threshold) & (img >= low_threshold))

        res[strong_i, strong_j] = np.int32(self.strong_pixel)
        res[weak_i, weak_j] = np.int32(self.weak_pixel)

        return res

    def hysteresis(self, img):
        hyst_len = 1
        res_img = img.copy()
        for y in range(hyst_len, res_img.shape[0] - hyst_len):
            for x in range(hyst_len, res_img.shape[1] - hyst_len):
                if res_img[y, x] == self.weak_pixel:
                    if ((res_img[y + hyst_len, x - hyst_len] == self.strong_pixel)
                            or (res_img[y + hyst_len, x] == self.strong_pixel)
                            or (res_img[y + hyst_len, x + hyst_len] == self.strong_pixel)
                            or (res_img[y, x - hyst_len] == self.strong_pixel)
                            or (res_img[y, x + hyst_len] == self.strong_pixel)
                            or (res_img[y - hyst_len, x - hyst_len] == self.strong_pixel)
                            or (res_img[y - hyst_len, x] == self.strong_pixel)
                            or (res_img[y - hyst_len, x + hyst_len] == self.strong_pixel)):
                        res_img[y, x] = self.strong_pixel
                    else:
                        res_img[y, x] = 0

        return res_img

    def detect(self, img):
        self.img_smoothed = full_conv(img, get_gaussian_kernel(self.kernel_size, self.sigma))
        self.grad_intens_matrix, self.grad_direct_matrix = self.sobel_filters(self.img_smoothed)
        self.non_max_img = self.non_max_suppression(self.grad_intens_matrix, self.grad_direct_matrix)
        self.threshold_img = self.threshold(self.non_max_img)
        self.hysteresis_img = self.hysteresis(self.threshold_img)

        return self.hysteresis_img


if __name__ == '__main__':
    detector = CannyEdgeDetector(
        sigma=1.5,
        kernel_size=5,
        lowthreshold=0.09,
        highthreshold=0.17,
        weak_pixel=75,
        strong_pixel=255
    )

    from matplotlib import pyplot as plt


    def visualize(img, title, show=True):
        plt.imshow(img, cmap='gray')
        plt.title(title)
        if show: plt.show()


    dir = 'images/'
    for img_name in os.listdir('images/'):
        img_orig = cv2.cvtColor(cv2.imread(f"{dir}{img_name}"), cv2.COLOR_BGR2GRAY)
        visualize(img_orig, img_name)

        detector.detect(np.array(img_orig, dtype='float64'))
        visualize(detector.img_smoothed, '1: img smoothed')
        visualize(detector.grad_intens_matrix, '2.1: gradient intensity matrix')
        visualize(detector.grad_direct_matrix, '2.2: gradient direction matrix')
        visualize(detector.non_max_img, '3: non maximum suppression')
        visualize(detector.threshold_img, '4: threshold')
        visualize(detector.hysteresis_img, '5: hysteresis(result)')

        # edges_cv = cv2.Canny(np.array(img_orig, dtype='uint8'), 75, 150)
        # visualize(edges_cv, 'Opencv Edges')
