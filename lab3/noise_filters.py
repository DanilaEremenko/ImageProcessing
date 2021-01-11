import math
import numpy as np
from numba import jit


######################################################################
# -------------------- BILATERAL NP ----------------------------------
######################################################################
@jit(nopython=True)
def distance_np(x1, y1, x2, y2):
    return np.sqrt(np.abs((x1 - x2) ** 2 + (y1 - y2) ** 2))


@jit(nopython=True)
def get_bilateral_kernel_np(small_img_arr, sigma_i, sigma_s):
    # assert len(small_img_arr.shape) == 2
    # assert small_img_arr.shape[0] == small_img_arr.shape[1]
    kernel_len = len(small_img_arr)
    center = small_img_arr.shape[0] // 2

    # np.meshgrid does't work with numba jit
    x_arr = np.zeros(shape=small_img_arr.shape)
    y_arr = np.zeros(shape=small_img_arr.shape)
    for x in range(kernel_len):
        for y in range(kernel_len):
            x_arr[x, y] = x
            y_arr[x, y] = y

    gi = gaussian_np(small_img_arr - small_img_arr[center, center], sigma_i)
    gs = gaussian_np(distance_np(x_arr, y_arr, center, center), sigma_s)
    kernel = gi * gs
    return kernel / kernel.sum()


######################################################################
# -------------------- BILATERAL PY ----------------------------------
######################################################################
@jit(nopython=True)
def distance_py(x1, y1, x2, y2):
    return math.sqrt(abs((x1 - x2) ** 2 + (y1 - y2) ** 2))


@jit(nopython=True)
def get_bilateral_kernel_py(small_img_arr, sigma_i, sigma_s):
    center = small_img_arr.shape[0] // 2
    kernel = np.zeros(shape=small_img_arr.shape)

    for x in range(len(small_img_arr)):
        for y in range(len(small_img_arr)):
            gi = gaussian_py(small_img_arr[x, y] - small_img_arr[center, center], sigma_i)
            gs = gaussian_py(distance_py(x, y, center, center), sigma_s)
            kernel[x, y] = gi * gs
    return kernel / kernel.sum()


######################################################################
# -------------------- GAUSSIAN_NP -----------------------------------
######################################################################
@jit(nopython=True)
def gaussian_np(x, sigma):
    return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-1 / 2 * ((x / sigma) ** 2))


@jit(nopython=True)
def get_gaussian_kernel_np(kernel_len, sigma):
    distance_vector = np.linspace(-(kernel_len - 1) / 2., (kernel_len - 1) / 2., kernel_len)
    kernel_1d = gaussian_np(distance_vector, sigma=sigma)
    kernel = np.outer(kernel_1d, kernel_1d)
    return kernel / np.sum(kernel)


######################################################################
# -------------------- GAUSSIAN_PY -----------------------------------
######################################################################
@jit(nopython=True)
def gaussian_py(x, sigma):
    return (1.0 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-1 / 2 * (x / sigma) ** 2)


@jit(nopython=True)
def get_gaussian_kernel_py(kernel_len, sigma):
    center = kernel_len // 2
    kernel = np.zeros((kernel_len, kernel_len))
    for x in range(kernel_len):
        for y in range(kernel_len):
            diff = distance_py(x, y, center, center)
            kernel[x, y] = gaussian_py(diff, sigma=sigma)
    return kernel / np.sum(kernel)


######################################################################
# -------------------- NON-LOCAL MEANS -------------------------------
######################################################################
def non_local_means(noisy, bw_size, sw_size, sigma, h, verbose=True):
    extd_img = np.pad(noisy, bw_size, mode='reflect')
    return non_local_means_numba(
        src_shape=noisy.shape,
        extd_img=extd_img,
        bw_size=bw_size,
        sw_size=sw_size,
        sigma=sigma,
        h=h,
        verbose=verbose
    )


@jit(nopython=True)
def non_local_means_numba(src_shape, extd_img, bw_size, sw_size, sigma, h, verbose=True):
    output_image = np.zeros((src_shape[0], src_shape[1]))
    extd_h, extd_w = extd_img.shape

    total_iterations = src_shape[0] * src_shape[1] * (2 * bw_size - sw_size) ** 2
    i = 0
    # big cycle
    for y in range(bw_size, extd_h - bw_size):
        for x in range(bw_size, extd_w - bw_size):
            # calculate weight using difference between neighbours
            comp_nbhd = extd_img[y:y + sw_size, x:x + sw_size]
            pixel_color = 0
            total_weight = 0
            for ypix in range(y - bw_size, y + bw_size - sw_size):
                for xpix in range(x - bw_size, x + bw_size - sw_size):
                    # создаем для текущего пикселя окна для рассчета gauss - L2 norm
                    curr_nbhd = extd_img[ypix:ypix + sw_size, xpix:xpix + sw_size]

                    # подсчет весов для текущего пикселя
                    distance = math.sqrt(np.sum(np.square(curr_nbhd - comp_nbhd)))
                    weight = math.exp(-(distance ** 2 - 2 * sigma ** 2) / h ** 2)
                    total_weight += weight
                    pixel_color += weight * extd_img[ypix, xpix]

                    # verbose part
                    i += 1
                    if verbose:
                        percent_complete = i / total_iterations * 100
                        if percent_complete % 5 == 0:
                            print('% COMPLETE = ', percent_complete)

            # обновляем изображения с учетом нового веса
            output_image[y - bw_size, x - bw_size] = pixel_color / total_weight

    return output_image
