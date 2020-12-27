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
def distance_py(x1, y1, x2, y2):
    return math.sqrt(abs((x1 - x2) ** 2 + (y1 - y2) ** 2))


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


def get_gaussian_kernel_np(kernel_len, sigma):
    distance_vector = np.linspace(-(kernel_len - 1) / 2., (kernel_len - 1) / 2., kernel_len)
    kernel_1d = gaussian_np(distance_vector, sigma=sigma)
    kernel = np.outer(kernel_1d, kernel_1d)
    return kernel / np.sum(kernel)


######################################################################
# -------------------- GAUSSIAN_PY -----------------------------------
######################################################################
def gaussian_py(x, sigma):
    return (1.0 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-1 / 2 * (x / sigma) ** 2)


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
def non_local_means(noisy, big_window_size, small_window_size, sigma, h, verbose=True):
    padwidth = big_window_size // 2
    extd_img = np.pad(noisy, ((padwidth, padwidth), (padwidth, padwidth)), mode='reflect')
    return non_local_means_numba(
        noisy=noisy,
        extd_img=extd_img,
        big_window_size=big_window_size,
        small_window_size=small_window_size,
        sigma=sigma,
        h=h,
        verbose=verbose
    )


@jit(nopython=True)
def non_local_means_numba(noisy, extd_img, big_window_size, small_window_size, sigma, h, verbose=True):
    """
    Performs the non-local-means algorithm given a noisy image.
    params is a tuple with:
    params = (big_window_size, small_window_size, h)
    Please keep big_window_size and small_window_size as even numbers
    """

    pad_width = big_window_size // 2
    img = noisy.copy()
    # ----------------------------------------------------------------------
    # ---------------- image extending yo bitch ----------------------------
    # ----------------------------------------------------------------------
    iterator = 0
    total_iterations = img.shape[1] * img.shape[0] * (big_window_size - small_window_size) ** 2

    if verbose: print("TOTAL ITERATIONS = ", total_iterations)

    output_image = extd_img.copy()  # TODO why not noisy
    small_center = small_window_size // 2

    # For each pixel in the source image, find a area around the pixel that needs to be compared
    for image_x in range(pad_width, pad_width + img.shape[1]):
        for image_y in range(pad_width, pad_width + img.shape[0]):

            b_win_x = image_x - pad_width
            b_win_y = image_y - pad_width

            # comparison neighbourhood
            comp_nbhd = extd_img[image_y - small_center:image_y + small_center + 1,
                        image_x - small_center:image_x + small_center + 1]

            pixel_color = 0
            total_weight = 0

            # For each comparison neighbourhood, search for all small windows within a large box, and compute their weights
            for s_win_x in range(b_win_x, b_win_x + big_window_size - small_window_size, 1):
                for s_win_y in range(b_win_y, b_win_y + big_window_size - small_window_size, 1):

                    # find the small box
                    curr_nbhd = extd_img[s_win_y:s_win_y + small_window_size + 1,
                                s_win_x:s_win_x + small_window_size + 1]

                    # weight is computed as a weighted softmax over the euclidean distances
                    distance = math.sqrt(np.sum(np.square(curr_nbhd - comp_nbhd)))
                    weight = math.exp(-(distance ** 2 - 2 * sigma ** 2) / h ** 2)
                    total_weight += weight
                    pixel_color += weight * extd_img[s_win_y + small_center, s_win_x + small_center]

                    # verbose part
                    iterator += 1
                    if verbose:
                        percent_complete = iterator * 100 / total_iterations
                        if percent_complete % 5 == 0:
                            print('% COMPLETE = ', percent_complete)

            pixel_color /= total_weight
            output_image[image_y, image_x] = pixel_color

    return output_image[pad_width:pad_width + img.shape[0], pad_width:pad_width + img.shape[1]]
