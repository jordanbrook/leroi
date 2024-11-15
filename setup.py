"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
from os import path
from io import open

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="leroi",
    version="0.0.1",
    description="Radar Data Objective Analysis Using Cressman Interpolation with a Leise Filter.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jordanbrook/leroi",
    author="Jordan Brook",    
    # For a list of valid classifiers, see https://pypi.org/classifiers/
    classifiers=[  # Optional
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    keywords="radar weather meteorology correction",
    packages=find_packages(exclude=["notebook"]),
    install_requires=["numpy", "arm_pyart", "astropy", "scipy"],
    project_urls={"Source": "https://github.com/jordanbrook/leroi/",},
)
