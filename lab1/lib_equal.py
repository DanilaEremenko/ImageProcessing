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
