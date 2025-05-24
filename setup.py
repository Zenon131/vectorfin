from setuptools import setup, find_packages

setup(
    name="vectorfin",
    version="0.1.0",
    packages=find_packages(),
    description="A multimodal financial analysis system combining text and market data",
    author="Jonathan Wallace",
    author_email="jonathan@example.com",
    install_requires=[
        "torch",
        "pandas",
        "numpy",
        "matplotlib",
        "transformers",
        "scikit-learn",
        "yfinance"
    ],
)
