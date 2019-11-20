import os
from setuptools import setup, find_packages

version = '0.0.1'

with open('requirements.txt') as f:
    install_requires = f.readlines()

setup(
    name='candle',
    version=version,
    packages=find_packages(),
    description='candle apps',
    install_requires=install_requires,
    python_requires=">=3.6.*",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering"
    ],
    keywords=[
        "FaaS",
    ],
    author='Globus labs',
    author_email='labs@globus.org',
    license="Apache License, Version 2.0",
    url="https://github.com/funcx-faas/funcx"
)
