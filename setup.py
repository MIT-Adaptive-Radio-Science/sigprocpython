#!/usr/bin/env python
"""
setup.py
This is the setup file for the ARS signal processing python package

@author: John Swoboda
"""
from pathlib import Path
from setuptools import setup, find_packages
import versioneer

req = ["scipy", "numpy", "matplotlib", "digital_rf"]
scripts = []


config = dict(
    description="Processing and Plotting of ",
    author="John Swoboda",
    url="https://github.com/MIT-Adaptive-Radio-Science/sigprocpython",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    install_requires=req,
    python_requires=">=3.0",
    packages=find_packages(),
    scripts=scripts,
    name="mitarspysigproc",
    package_data={"mitarspysigproc": ["coeffs/*.csv"]},
)

curpath = Path(__file__)
testpath = curpath.joinpath("Testdata")
try:
    curpath.mkdir(parents=True, exist_ok=True)
    print("created {}".format(testpath))
except OSError:
    pass


setup(**config)
