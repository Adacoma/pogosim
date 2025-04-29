#!/usr/bin/env python3
from setuptools import setup, find_packages
import pogosim

setup(
    name="pogosim",
    version=pogosim.__version__,
    description="Pogobot Batch Runner",
    long_description=open("../README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Leo Cazenille",
    author_email="leo.cazenille@gmail.com",
    url="https://github.com/Adacoma/pogosim",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "pogobatch = pogosim.pogobatch:main",
        ],
    },
    install_requires=[
        "pyyaml>=5.3",
        "pandas>=1.3.0",
        "pyarrow>=19.0.0",
        # "ray>=2.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)

# MODELINE "{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
