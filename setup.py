from setuptools import setup, find_packages

setup(name='aegis',
  version='0.1.0',
  install_requires = [
    'ml_utils @ git+https://github.com/tehzevo/ml-utils@master#egg=ml_utils',
    'pget @ git+https://github.com/tehzevo/pget@master#egg=pget',
    'tensorflow',
    'numpy',
    'flask',
    'flask_restful',
    'requests'
  ],
  packages=find_packages())
