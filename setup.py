from distutils.core import setup, Extension

approx_knn_c = Extension('approx_knn_c.approx_knn',
        language = 'c++',
        extra_compile_args = ["-std=c++11", "-O3", '-pthread', "-lm"],
        extra_link_args = ['-lm'],
        include_dirs = ['.'],
        sources = ['approx_knn_c/approx_knn.cpp'],
        depends = ['approx_knn_c/annoylib.h', 'approx_knn_c/kissrandom.h'])

setup (name = 'approx_knn_c',
       version = '0.1',
       url = '',
       author = '',
       author_email = '',
       license = '',
       description = 'K-NN search',
       packages=['approx_knn_c'],
       ext_modules = [approx_knn_c]
)
