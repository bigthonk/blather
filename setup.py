#!/usr/bin/env python

from setuptools import setup

setup(
    name='blather',
    version='1.0.7',
    packages=['blather'],
    install_requires=[
        'torch',
        'transformers',
    ],
)
