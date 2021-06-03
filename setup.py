# -*- coding: utf-8 -*-
"""
Created on Thu May 20 17:06:19 2021

@author: Dang Trung Anh
"""

import os
from pip._internal.network.session import PipSession
from setuptools import setup, find_packages

VERSION = '1.0.4' 
DESCRIPTION = 'An Open-Source Nature-Inspired Optimization Clustering Framework in Python'
# LONGDESCRIPTION = 'EvoCluster is an open source and cross-platform framework implemented in Python which includes the most well-known and recent nature-inspired meta heuristic optimizers that are customized to perform partitional clustering tasks. The goal of this framework is to provide a user-friendly and customizable implementation of the metaheuristic based clustering algorithms which canbe utilized by experienced and non-experienced users for different applications.The framework can also be used by researchers who can benefit from the implementation of the metaheuristic optimizers for their research studies. EvoClustercan be extended by designing other optimizers, including more objective func-tions, adding other evaluation measures, and using more data sets. The current implementation of the framework includes ten metaheuristic optimizers, thirty datasets, five objective functions, twelve evaluation measures, more than twenty distance measures, and ten different ways for detecting the k value.'

from pip._internal.req import parse_requirements

install_reqs = parse_requirements('requirements.txt', session=PipSession())
reqs = [str(ir.requirement) for ir in install_reqs]

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'long_description.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
       # the name must match the folder name 'EvoCC'
        name="EvoCluster", 
        version=VERSION,
        author="Raneem Qaddoura, Hossam Faris, Ibrahim Aljarah, and Pedro A. Castillo",
        author_email="dangtrunganh@gmail.com, raneem.qaddoura@gmail.com",
        description=DESCRIPTION,
        long_description=long_description,
        long_description_content_type='text/markdown',
        # package_dir={"": "EvoCluster"},
        packages=find_packages(),
        # add any additional packages that 
        install_requires=reqs,
        url='https://github.com/RaneemQaddoura/EvoCluster',
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
