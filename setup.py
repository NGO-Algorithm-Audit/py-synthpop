from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="python-synthpop",
    version="0.1.0",
    author="Algorithm Audit",
    description="Python implementation of the R package synthpop for generating synthetic data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/algorithm-audit/python-synthpop",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
        ],
    },
)
