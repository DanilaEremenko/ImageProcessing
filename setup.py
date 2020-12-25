from setuptools import setup
from Cython.Build import cythonize

# python setup.py build_ext --inplace

setup(
    ext_modules=cythonize(
        "lab2/conv_cython.pyx",
        "lab3/noise_filters_cython.pyx"
    )
)
