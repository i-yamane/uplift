#!/usr/bin/env python

from setuptools import setup, find_packages
from codecs import open
from os import path

path2readme = path.abspath(path.dirname(__file__))
with open(path.join(path2readme, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='uplift',
    version='0.3.1',
    description='Code for Uplift Modeling.',
    long_description=long_description,
    url='https://github.com/i-yamane/uplift',
    author='Ikko Yamane',
    author_email='yamane@ms.k.u-tokyo.ac.jp',
    python_requires='>=3',
    packages=find_packages(exclude=[])
)
