"""
==========================================================
HQFSF Setup Script
==========================================================

Hybrid Quantum Feature Selection Framework
Using Variational Quantum Circuits

Author  : Jasmine Sultana
Version : 1.0.0
License : MIT
==========================================================
"""

from pathlib import Path
from setuptools import setup, find_packages

BASE_DIR = Path(__file__).parent

README = (BASE_DIR / "README.md").read_text(encoding="utf-8")
REQUIREMENTS = [
    line.strip()
    for line in (BASE_DIR / "requirements.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    if line.strip() and not line.startswith("#")
]

setup(
    name="hqfsf",
    version="1.0.0",
    author="Jasmine Sultana",
    author_email="your_email@example.com",

    description="Hybrid Quantum Feature Selection Framework Using Variational Quantum Circuits",

    long_description=README,
    long_description_content_type="text/markdown",

    url="https://github.com/Jasy002/HQFSF",

    packages=find_packages(
        exclude=[
            "tests",
            "docs",
            "examples",
        ]
    ),

    include_package_data=True,

    install_requires=REQUIREMENTS,

    python_requires=">=3.11",

    license="MIT",

    keywords=[
        "Quantum Computing",
        "Feature Selection",
        "Machine Learning",
        "Quantum Machine Learning",
        "Qiskit",
        "Artificial Intelligence",
    ],

    project_urls={
        "Source": "https://github.com/Jasy002/HQFSF",
        "Issues": "https://github.com/Jasy002/HQFSF/issues",
    },

    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)