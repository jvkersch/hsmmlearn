hsmmlearn
=========

[![Linux build Status](https://travis-ci.org/jvkersch/hsmmlearn.svg?branch=master)](https://travis-ci.org/jvkersch/hsmmlearn)
[![Windows build status](https://ci.appveyor.com/api/projects/status/elnatei2kavchwg1/branch/master?svg=true)](https://ci.appveyor.com/project/jvkersch/hsmmlearn)

[![Coverage Status](https://coveralls.io/repos/github/jvkersch/hsmmlearn/badge.svg?branch=master)](https://coveralls.io/github/jvkersch/hsmmlearn?branch=master)

`hsmmlearn` is a library for **unsupervised** learning of hidden semi-Markov
models with explicit durations. It is a port of the
[hsmm package](https://cran.r-project.org/web/packages/hsmm/) for R, and in
fact wraps the same underlying C++ library.

`hsmmlearn` borrows its name and the design of its api from
[hmmlearn](http://hmmlearn.readthedocs.org/en/latest/).

Install
-------

`hsmmlearn` supports Python 2.7 and Python 3.4 and up. After cloning the
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

To run the unit tests, do
```console
python -m unittest discover -v .
```

Building the documentation
--------------------------

The documentation for `hsmmlearn` is a work in progress. To build the docs,
first install the doc requirements, then run Sphinx:
```console
cd docs
pip install -r doc_requirements.txt
make html
```
If everything goes well, the documentation should be in `docs/_build/html`.

Some of the documentation comes as jupyter notebooks, which can be found in the
`notebooks/` folder. Sphinx ingests these, and produces rst documents out of
them. If you end up modifying the notebooks, run `make notebooks` in the
documentation folder and check in the output.

License
-------

hsmmlearn incorporates a significant amount of code from R's hsmm package, and
is therefore released under the
[GPL, version 3.0](http://www.gnu.org/licenses/gpl-3.0.en.html).
