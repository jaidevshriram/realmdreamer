from setuptools import setup
from Cython.Build import cythonize

setup(
    name="occlude",
    version="0.1.0",
    ext_modules=cythonize("occlude.pyx"),
    install_requires=[
        'cython',
    ],
)