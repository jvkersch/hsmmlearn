hsmmlearn
=========

[![Linux build Status](https://travis-ci.org/jvkersch/hsmmlearn.svg?branch=master)](https://travis-ci.org/jvkersch/hsmmlearn)
[![Windows build status](https://ci.appveyor.com/api/projects/status/elnatei2kavchwg1/branch/master?svg=true)](https://ci.appveyor.com/project/jvkersch/hsmmlearn)

hsmmlearn is a library for **unsupervised** learning of hidden semi-Markov
models with explicit durations. It is a port of the
[hsmm package](https://cran.r-project.org/web/packages/hsmm/) for R, and in
fact wraps the same underlying C++ library.

hsmmlearn borrows its name and the design of its api from
[hmmlearn](http://hmmlearn.readthedocs.org/en/latest/).

Install
-------

hsmmlearn supports Python 2.7 and Python 3.4 and up. After cloning the
repository, first install the requirements
```bash
pip install -r requirements.txt
```
Then run either
```bash
python setup.py develop
```
or
```bash
python setup.py install
```
to install the package from source.

License
-------

hsmmlearn incorporates a significant amount of code from R's hsmm package, and
is therefore released under the
[GPL, version 3.0](http://www.gnu.org/licenses/gpl-3.0.en.html).
