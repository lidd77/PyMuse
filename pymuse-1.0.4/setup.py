#!/usr/bin/env python
from setuptools import setup

setup(
    name='pymuse',
    version='1.0.4',
    description="pymuse-A wrapped  pymuse class",
    author="lidongdong",
    license="LICENSE",
    scripts=['pymuse.py'],
    install_requires=['pandas >= 0.22.0','numpy >= 1.14.2'],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Environment :: Console",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX",
        "Programming Language :: Python",
        "Topic :: Text Processing"
    ],
)

