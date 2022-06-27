#!/usr/bin/env python
"""
setup.py
This is the setup file for the VSRTProc python package

@author: John Swoboda
"""
from pathlib import Path
from setuptools import setup, find_packages
import versioneer

req = ["scipy","matplotlib"]
scripts = ["bin/makemicplots.py","bin/rundetector.py"]


config = dict(
    description="Processing and Plotting of ",
    author="John Swoboda",
    url="https://github.mit.edu/MOXIE/MarsMicAnalysis",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    install_requires=req,
    python_requires=">=3.0",
    packages=find_packages(),
    scripts=scripts,
    name="moxiemictools",
)

curpath = Path(__file__)
testpath = curpath.joinpath("Testdata")
try:
    curpath.mkdir(parents=True, exist_ok=True)
    print("created {}".format(testpath))
except OSError:
    pass


setup(**config)
