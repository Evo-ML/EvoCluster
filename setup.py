from setuptools import setup, find_packages

setup(
    name="EvoCluster",
    version="1.0.0",
    author="Raneem Qaddoura, Hossam Faris, and Ibrahim Aljarah",
    author_email="raneem.qaddoura@gmail.com",
    description="An Open-Source Nature-Inspired Optimization Clustering Framework in Python",
    long_description="",
    long_description_content_type="text/markdown",
    url="https://github.com/RaneemQaddoura/EvoCluster",
    packages=find_packages(where='src'),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License 2.0",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)