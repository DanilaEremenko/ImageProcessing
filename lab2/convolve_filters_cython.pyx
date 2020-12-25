def conv_cython(img_sub_arr, kernel):
    kernel_len = len(img_sub_arr)
    sum = 0
    for x in range(kernel_len):
        for y in range(kernel_len):
            sum += img_sub_arr[x, y] * kernel[x, y]
    return sum
