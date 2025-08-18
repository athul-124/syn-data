"""
Setup script for SynData Plus
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="syndata-plus",
    version="1.0.0",
    author="SynData Plus Team",
    author_email="team@syndataplus.com",
    description="Advanced synthetic data generator using SDV + Faker",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/syndataplus/syndata-plus",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "sdv>=1.6.0",
        "faker>=19.0.0",
        "click>=8.1.0",
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "python-multipart>=0.0.6",
        "rich>=13.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "syndata-plus=syndata_plus.cli:cli",
        ],
    },
)
