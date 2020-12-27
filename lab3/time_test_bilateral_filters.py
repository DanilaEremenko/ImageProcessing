from noise_filters_cython import get_bilateral_kernel_cython
from noise_filters import get_bilateral_kernel_np, get_bilateral_kernel_py

import time
import numpy as np

sigma_i = 5
sigma_s = 10
arr = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])

compare_dict = {
    'cython': {'func': get_bilateral_kernel_cython},
    'python_np': {'func': get_bilateral_kernel_np},
    'python_py': {'func': get_bilateral_kernel_py}
}

CALL_NUM = 100_000

# run time tests
for key, value in compare_dict.items():
    start_time = time.time()
    for i in range(CALL_NUM):
        compare_dict[key]['res'] = compare_dict[key]['func'](small_img_arr=arr, sigma_i=sigma_i, sigma_s=sigma_s)
    print(f"{key} time = {time.time() - start_time}")

print("-----COMPARE RESULTS--------")
# round results for equal tests
for key, value in compare_dict.items():
    compare_dict[key]['res'] = compare_dict[key]['res'].round(3)

for i in range(len(compare_dict.keys()) - 1):
    curr_key = list(compare_dict.keys())[i]
    next_key = list(compare_dict.keys())[i + 1]
    if (compare_dict[curr_key]['res'] == compare_dict[next_key]['res']).all():
        print(f'results {curr_key} == {next_key}')

    else:
        print(f'results {curr_key} != {next_key}\n')
        print(f"kernel {curr_key} =\n {str(compare_dict[curr_key]['res'])}\n")
        print(f"kernel {next_key} =\n {str(compare_dict[next_key]['res'])}")
    print('--')
