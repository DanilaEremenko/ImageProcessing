from convolve_filters_cython import conv_cython
from convolve_filters import conv_np, conv_py
import numpy as np

from test_lib.test_lib import compare_functions

args = {'arr': np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]]), 'kernel': np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])}
compare_functions(
    test_name='CONVOLUTION',
    compare_dict={
        'cython': {'func': conv_cython, 'args': args},
        'python_np': {'func': conv_np, 'args': args},
        'python_py': {'func': conv_py, 'args': args}
    },
    call_num=100_000
)
