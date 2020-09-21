from distutils.core import setup
from Cython.Build import cythonize
import numpy
                                    
setup(
  name = 'plrmm',
  version = '0.1',
  packages = ['plrmm'],
  ext_modules = cythonize("plrmm/plfunc.pyx"),
  include_dirs = [numpy.get_include()],
  scripts = ['bin/plrmm_train.py', 'bin/plrmm_predz.py', 'bin/plrmm_predy.py'],
)