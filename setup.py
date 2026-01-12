"""
Setup file for ml_multiomics package.

Installation:
    pip install -e .

Or for development:
    pip install -e .[dev]
"""

from setuptools import setup, find_packages
import pathlib

# Read the README file
here = pathlib.Path(__file__).parent.resolve()
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='ml_multiomics',
    version='0.1.0',
    description='Multi-omics machine learning framework for integrative analysis',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Your Name',
    author_email='your.email@example.com',
    
    # Package discovery
    packages=find_packages(),
    
    # Python version requirement
    python_requires='>=3.8',
    
    # Dependencies
    install_requires=[
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'matplotlib>=3.4.0',
        'seaborn>=0.11.0',
        'scikit-learn>=1.0.0',
        'scipy>=1.7.0',
    ],
    
    # Optional dependencies for development
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'black>=21.0',
            'flake8>=3.9',
            'jupyter>=1.0',
            'ipython>=7.0',
        ],
    },
    
    # Package classifiers
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    
    # Keywords
    keywords='omics, multi-omics, machine-learning, metabolomics, proteomics, integration',
    
    # Project URLs
    project_urls={
        'Bug Reports': 'https://github.com/yourusername/ml_multiomics/issues',
        'Source': 'https://github.com/yourusername/ml_multiomics',
    },
)