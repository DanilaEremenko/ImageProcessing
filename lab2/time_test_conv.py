from conv_cython import conv_cython
from convolve_filters import conv
import time
import numpy as np

arr = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
kernel = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])

compare_dict = {
    'cython': {'func': conv_cython},
    'python_np': {'func': conv}
}

CALL_NUM = 100_000

# run time tests
for key, value in compare_dict.items():
    start_time = time.time()
    for i in range(CALL_NUM):
        compare_dict[key]['res'] = compare_dict[key]['func'](arr, kernel)
    print(f"{key} time = {time.time() - start_time}")

# round results for equal tests
for key, value in compare_dict.items():
    for i in range(CALL_NUM):
        compare_dict[key]['res'] = round(compare_dict[key]['res'], 5)

if (compare_dict['cython']['res'] == compare_dict['python_np']['res']).all():
    print('results equal')
else:
    print('results is not equal')
