from noise_filters_cython import get_bilateral_kernel_cython
from noise_filters import get_bilateral_kernel_np, get_bilateral_kernel_py

import numpy as np

from test_lib.test_lib import compare_functions

args = {'sigma_i': 5, 'sigma_s': 10, 'small_img_arr': np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])}
compare_functions(
    test_name='BILATERAL FILTERS',
    compare_dict={
        'cython': {'func': get_bilateral_kernel_cython, 'args': args},
        'python_np': {'func': get_bilateral_kernel_np, 'args': args},
        'python_py': {'func': get_bilateral_kernel_py, 'args': args}
    },
    call_num=100_000
)
