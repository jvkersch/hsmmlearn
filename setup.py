""" Hidden semi-Markov models with explicit durations in Python.
"""
import glob

import setuptools  # noqa

from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

sources = (
    ["simple_hsmm/base.pyx"] + glob.glob('simple_hsmm/_hsmm/src/*.cpp')
)
extensions = [
    Extension("simple_hsmm.base", sources, language="c++")
]

import simple_hsmm

CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "License :: OSI Approved",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "Topic :: Software Development",
    "Topic :: Scientific/Engineering",
    "Programming Language :: Cython",
    "Programming Language :: Python",
    "Programming Language :: Python :: 2",
    "Programming Language :: Python :: 2.7",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.4",
    "Programming Language :: Python :: 3.5",
]

DESCRIPTION = __doc__
LONG_DESCRIPTION = open('README.md').read()
MAINTAINER = 'Joris Vankerschaver'
MAINTAINER_EMAIL = 'Joris.Vankerschaver@gmail.com'
LICENSE = 'GPL v3'

VERSION = simple_hsmm._version

setup(
    name='simple_hsmm',
    ext_modules=cythonize(extensions),
    packages=['simple_hsmm'],
    package_data={'simple_hsmm': ['base.pyx']},
    classifiers=CLASSIFIERS,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    license=LICENSE,
)
