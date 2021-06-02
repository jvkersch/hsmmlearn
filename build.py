from Cython.Build import cythonize
from distutils.command.build_ext import build_ext
from distutils.extension import Extension
import glob


def build(setup_kwargs):
    sources = (["hsmmlearn/base.pyx"] + glob.glob('hsmmlearn/_hsmm/src/*.cpp'))
    extensions = [Extension("hsmmlearn.base", sources, language="c++")]
    setup_kwargs.update({
        'ext_modules':
        cythonize(
            extensions,
            language_level=3,
            compiler_directives={'linetrace': True},
        ),
        'cmdclass': {
            'build_ext': build_ext
        }
    })
