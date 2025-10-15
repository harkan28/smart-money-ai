#!/usr/bin/env python3
"""
Smart Money AI - Setup Configuration
===================================

Professional setup configuration for Smart Money AI package
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
README_PATH = Path(__file__).parent / "README.md"
long_description = README_PATH.read_text(encoding="utf-8") if README_PATH.exists() else ""

# Read requirements
REQUIREMENTS_PATH = Path(__file__).parent / "requirements-production.txt"
requirements = []
if REQUIREMENTS_PATH.exists():
    requirements = [
        line.strip() for line in REQUIREMENTS_PATH.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]

# Development requirements
dev_requirements = [
    "pytest>=7.1.0",
    "pytest-cov>=3.0.0",
    "black>=22.6.0",
    "flake8>=5.0.0",
    "mypy>=0.971",
    "isort>=5.10.0",
    "pre-commit>=2.20.0",
    "jupyter>=1.0.0",
    "notebook>=6.4.0"
]

# Optional requirements for different features
ml_requirements = [
    "torch>=1.12.0",
    "transformers>=4.20.0",
    "sentence-transformers>=2.2.0",
    "tensorflow>=2.9.0"
]

api_requirements = [
    "fastapi>=0.85.0",
    "uvicorn>=0.18.0",
    "python-multipart>=0.0.5",
    "gunicorn>=20.1.0"
]

viz_requirements = [
    "matplotlib>=3.5.0",
    "seaborn>=0.11.0",
    "plotly>=5.10.0",
    "dash>=2.6.0"
]

setup(
    name="smart-money-ai",
    version="2.0.0",
    author="Smart Money AI Team",
    author_email="contact@smartmoneyai.com",
    description="Intelligent Financial Management System with AI-powered insights",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/harkan28/smart-money-ai",
    project_urls={
        "Bug Reports": "https://github.com/harkan28/smart-money-ai/issues",
        "Source": "https://github.com/harkan28/smart-money-ai",
        "Documentation": "https://smart-money-ai.readthedocs.io/",
    },
    
    # Package discovery
    packages=find_packages(include=["src", "src.*"]),
    package_dir={"": "."},
    
    # Include package data
    include_package_data=True,
    package_data={
        "src": [
            "**/*.json",
            "**/*.yaml",
            "**/*.yml",
            "**/*.txt",
            "**/*.md"
        ],
    },
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Dependencies
    install_requires=requirements,
    
    # Optional dependencies
    extras_require={
        "dev": dev_requirements,
        "ml": ml_requirements,
        "api": api_requirements,
        "viz": viz_requirements,
        "all": dev_requirements + ml_requirements + api_requirements + viz_requirements,
    },
    
    # Entry points for command-line tools
    entry_points={
        "console_scripts": [
            "smart-money=src.cli.main:main",
            "smart-money-train=src.cli.train:main",
            "smart-money-api=src.api.server:main",
            "smart-money-test=tests.test_comprehensive:run_tests",
        ],
    },
    
    # Classifiers for PyPI
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    
    # Keywords for discoverability
    keywords=[
        "fintech", "ai", "machine-learning", "financial-analysis",
        "expense-tracking", "investment", "personal-finance",
        "sms-parsing", "behavioral-finance", "predictive-analytics"
    ],
    
    # Minimum versions for critical dependencies
    zip_safe=False,
    
    # Additional metadata
    platforms=["any"],
    license="MIT",
    
    # Testing
    test_suite="tests",
    tests_require=dev_requirements,
)