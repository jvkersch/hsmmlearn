import glob

import setuptools  # noqa

from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

sources = (
    ["simple_hsmm/base/_base.pyx"] + glob.glob('simple_hsmm/_hsmm/src/*.cpp')
)
extensions = [
    Extension("simple_hsmm._base", sources, language="c++")
]

setup(
    name='simple_hsmm',
    ext_modules=cythonize(extensions)
)
