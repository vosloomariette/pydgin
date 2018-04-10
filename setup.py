#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


if sys.argv[-1] == 'publish':
    os.system('python setup.py sdist upload')
    sys.exit()


packages = [
    'pydgin',
]

package_data = {
}

requires = [
    'numpy', 'numba',
]

extra_requires = {
    'ipy': ['ipywidgets'],
}

classifiers = [
]

setup(
    name='pydgin',
    version='0.0.5',
    description='',
    long_description='',
    packages=packages,
    package_data=package_data,
    install_requires=requires,
    extra_requires=extra_requires,
    url='',
    license='MIT',
    classifiers=classifiers,
    author='Mariette Vosloo',
    author_email='vosloomariette@gmail.com',
)
