"%sdkverpath%" -q -version:"%sdkver%"
call setenv /x64

rem install openblas for scipy

mkdir Downloads
cd Downloads
appveyor DownloadFile http://downloads.sourceforge.net/project/openblas/v0.2.15/OpenBLAS-v0.2.15-Win64-int32.zip
7z e OpenBLAS-v0.2.15-Win64-int32.zip
dir
copy libopenblas.dll C:\
cd %APPVEYOR_BUILD_FOLDER%
dir C:\

rem install python packages
pip install --cache-dir C:/egg_cache nose
pip install --cache-dir C:/egg_cache coverage==3.7.1
pip install --cache-dir C:/egg_cache numpy
pip install --cache-dir C:/egg_cache scipy
pip install --cache-dir C:/egg_cache cython
pip install --cache-dir C:/egg_cache six

rem install package
python setup.py develop
