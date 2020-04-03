from setuptools import setup, find_packages

setup(name='aegis_core',
  version='0.7.0',
  install_requires = [
    'ml_utils @ git+https://github.com/tehzevo/ml-utils@master#egg=ml_utils',
    'tensorflow',
    'numpy',
    'flask',
    'flask_restful',
    'flask-cors',
    'requests',
    'opencv-python',
  ],
  packages=find_packages())
