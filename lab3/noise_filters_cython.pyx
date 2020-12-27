import numpy as np
import math

cdef extern from "math.h":
    float exp(float theta)
    float abs(float theta)
    float sqrt(float theta)

cdef float pi = math.pi

cdef gaussian(float x, float sigma):
    return (1.0 / (sigma * sqrt(2 * pi))) * exp(-1 / 2 * (x / sigma) ** 2)

cdef distance(float x1, float y1, float x2, float y2):
    return sqrt(abs((x1 - x2) ** 2 + (y1 - y2) ** 2))

def get_bilateral_kernel_cython(small_img_arr, float sigma_i, float sigma_s):
    cdef int kernel_len = len(small_img_arr)
    cdef Py_ssize_t x_max = small_img_arr.shape[0]
    cdef Py_ssize_t y_max = small_img_arr.shape[1]

    cdef int center = small_img_arr.shape[0] // 2
    kernel = np.zeros(shape=small_img_arr.shape)
    cdef float gi
    cdef float gs
    for x in range(x_max):
        for y in range(y_max):
            gi = gaussian(small_img_arr[x, y] - small_img_arr[center, center], sigma_i)
            gs = gaussian(distance(x, y, center, center), sigma_s)
            kernel[x, y] = gi * gs
    return kernel / kernel.sum()
