""" Hidden semi-Markov models with explicit durations in Python.
"""
import glob

import os

from setuptools import find_packages
from distutils.core import setup
from distutils.extension import Extension

SOURCES = (
    ["hsmmlearn/base.pyx"] + glob.glob('hsmmlearn/_hsmm/src/*.cpp')
)


def get_extension_modules():
    # ReadTheDocs has trouble with C extension modules, so don't build the
    # Cython modules.
    on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
    if on_rtd:
        return []
    else:
        from Cython.Build import cythonize
        extensions = [
            Extension("hsmmlearn.base", SOURCES, language="c++")
        ]
        return cythonize(extensions)


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

REQUIREMENTS = [
    "cython",
    "numpy",
    "six",
    "scipy",
]
    

DESCRIPTION = __doc__
LONG_DESCRIPTION = open('README.md').read()
MAINTAINER = 'Joris Vankerschaver'
MAINTAINER_EMAIL = 'Joris.Vankerschaver@gmail.com'
LICENSE = 'GPL v3'

setup(
    name='hsmmlearn',
    ext_modules=get_extension_modules(),
    packages=find_packages(include=["hsmmlearn", "hsmmlearn.*"]),
    include_package_data=True,
    install_requires=REQUIREMENTS,
    classifiers=CLASSIFIERS,
    version='0.1.0',
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    license=LICENSE,
)
