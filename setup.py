from setuptools import setup, find_packages

setup(name='aegis_core',
  version='0.8.0',
  install_requires = [
    'ml_utils @ git+https://github.com/tehzevo/ml-utils@master#egg=ml_utils',
    'dehydrated_vae @ git+https://github.com/tehzevo/dehydrated-vae@master#egg=dehydrated_vae',
    'tensorflow',
    'numpy',
    'flask',
    'flask_restful',
    'flask-cors',
    'requests',
    'opencv-python',
  ],
  packages=find_packages())
