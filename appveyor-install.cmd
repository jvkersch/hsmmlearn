"%sdkverpath%" -q -version:"%sdkver%"
call setenv /x64

rem install python packages
pip install --cache-dir C:/egg_cache nose
pip install --cache-dir C:/egg_cache coverage==3.7.1
pip install --cache-dir C:/egg_cache cython
pip install --cache-dir C:/egg_cache six
pip install -i https://pypi.binstar.org/carlkl/simple --cache-dir C:/egg_cache numpy
pip install -i https://pypi.binstar.org/carlkl/simple --cache-dir C:/egg_cache scipy

rem install package
python setup.py develop
