#!/usr/bin/env python3
"""
Setup script for Enhanced Crypto Volatility Analysis
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="crypto-volatility-analysis",
    version="2.0.0",
    author="Crypto Volatility Analysis Project",
    author_email="your-email@example.com",  # Replace with your email
    description="A comprehensive analysis tool to identify the best cryptocurrency for scalping/day trading",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/imanhavangi/crypto-volatility-analysis",  # Replace with your repo URL
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "crypto-volatility=main:main",
        ],
    },
    keywords="cryptocurrency, trading, volatility, analysis, scalping, fintech, ccxt",
    project_urls={
        "Bug Reports": "https://github.com/imanhavangi/crypto-volatility-analysis/issues",
        "Source": "https://github.com/imanhavangi/crypto-volatility-analysis",
        "Documentation": "https://github.com/imanhavangi/crypto-volatility-analysis#readme",
    },
) 