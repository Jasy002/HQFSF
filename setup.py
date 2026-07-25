from setuptools import setup, find_packages

setup(
    name="hqfsf",
    version="1.0.0",
    author="Jasmine Sultana",
    author_email="your_email@example.com",
    description="Hybrid Quantum Feature Selection Framework Using Variational Quantum Circuits",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Jasy002/HQFSF",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "matplotlib",
        "plotly",
        "qiskit",
        "qiskit-aer",
        "PyYAML",
        "joblib",
    ],
    python_requires=">=3.11",
    license="MIT",
)