"""Setup configuration for pod5-accelerator package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pod5-accelerator",
    version="0.1.0",
    author="Samuel Aboderin",
    author_email="aboderinseun01@gmail.com",
    description="High-performance POD5 file reader for Oxford Nanopore sequencing data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aboderinsamuel/pod5-accelerator",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pod5>=0.2.0",
        "pyarrow>=10.0.0",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "pytest>=7.0.0",
        "psutil>=5.9.0",
    ],
    extras_require={
        "dev": [
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
        ],
    },
    entry_points={
        "console_scripts": [
            "pod5-benchmark=pod5_accelerator.benchmarks.runner:main",
        ],
    },
)
