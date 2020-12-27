def conv_cython(arr, kernel):
    kernel_len = len(arr)
    sum = 0
    for x in range(kernel_len):
        for y in range(kernel_len):
            sum += arr[x, y] * kernel[x, y]
    return sum
