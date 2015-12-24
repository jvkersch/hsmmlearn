""" Hidden semi-Markov models with explicit durations in Python.
"""
import glob

import setuptools  # noqa

from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

sources = (
    ["hsmmlearn/base.pyx"] + glob.glob('hsmmlearn/_hsmm/src/*.cpp')
)
extensions = [
    Extension("hsmmlearn.base", sources, language="c++")
]

import hsmmlearn

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

VERSION = hsmmlearn._version

setup(
    name='hsmmlearn',
    ext_modules=cythonize(extensions),
    packages=['hsmmlearn'],
    package_data={'hsmmlearn': ['base.pyx']},
    classifiers=CLASSIFIERS,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    license=LICENSE,
)
