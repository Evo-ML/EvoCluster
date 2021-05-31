# -*- coding: utf-8 -*-
"""
Created on Thu May 20 17:06:19 2021

@author: Dang Trung Anh
"""

import os
from pip._internal.network.session import PipSession
from setuptools import setup, find_packages

VERSION = '1.0.1' 
DESCRIPTION = 'An Open-Source Nature-Inspired Optimization Clustering Framework in Python'
LONG_DESCRIPTION = 'An Open-Source Nature-Inspired Optimization Clustering Framework in Python'

from pip._internal.req import parse_requirements

install_reqs = parse_requirements('requirements.txt', session=PipSession())
reqs = [str(ir.requirement) for ir in install_reqs]

setup(
       # the name must match the folder name 'EvoCC'
        name="EvoCluster", 
        version=VERSION,
        author="Raneem Qaddoura, Hossam Faris, and Ibrahim Aljarah",
        author_email="dangtrunganh@gmail.com, raneem.qaddoura@gmail.com",
        description="An Open-Source Nature-Inspired Optimization Clustering Framework in Python",
        long_description=LONG_DESCRIPTION,
        # package_dir={"": "EvoCluster"},
        packages=find_packages(),
        # add any additional packages that 
        install_requires=reqs,
        url="https://github.com/RaneemQaddoura/EvoCluster",
        keywords=['python', 'first package'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ],
        package_data = {'': ['datasets/*.csv']},
        
)