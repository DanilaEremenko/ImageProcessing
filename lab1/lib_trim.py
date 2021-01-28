def build_change_matrix(new_left, new_right, left, right):
    # (i - a) * ((d - c) / (b - a)) + c
    new_range = new_right - new_left
    old_range = right - left
    range_multiplier = new_range / old_range
    result_matrix = []
    for i in range(new_range):
        new_color = int((i - left) * range_multiplier + new_left)
        result_matrix.append(max(min(new_color, 255), 0))
    return result_matrix


def trim_part(part, hist, left_pix=0, right_pix=255):
    res_hist = hist.copy()

    left_cut_n = sum(res_hist) * part // 2
    right_cut_n = left_cut_n

    print(f"{left_cut_n} - value to cut off from {sum(res_hist)}")

    while left_cut_n > 0:
        if hist[left_pix] < left_cut_n:
            left_cut_n -= res_hist[left_pix]
            res_hist[left_pix] = 0
            left_pix += 1
        else:
            res_hist[left_pix] -= left_cut_n
            left_cut_n = 0
    while right_cut_n > 0:
        if res_hist[right_pix] < right_cut_n:
            right_cut_n -= res_hist[right_pix]
            res_hist[right_pix] = 0
            right_pix -= 1
        else:
            res_hist[right_pix] -= right_cut_n
            right_cut_n = 0
    print("new borders: " + str(left_pix) + " - " + str(right_pix))
    return left_pix, right_pix, res_hist
