"""
Setup configuration for PharmaSentinel.
"""

from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="pharmasentinel",
    version="1.0.0",
    author="Harshita Adlakha",
    author_email="parasadlakha456@gmail.com",
    description=(
        "Explainable Multi-Task Clinical NLP Framework for Drug Efficacy Assessment"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/HarshitaAdlakha/pharmasentinel",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.38.0",
        "scikit-learn>=1.4.0",
        "pandas>=2.1.0",
        "numpy>=1.26.0",
        "plotly>=5.18.0",
        "streamlit>=1.32.0",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
        "shap>=0.44.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="nlp bert drug-recommendation explainable-ai multi-task-learning clinical-nlp",
    entry_points={
        "console_scripts": [
            "pharmasentinel-train=scripts.train_bert:main",
            "pharmasentinel-baseline=scripts.train_baseline:main",
        ],
    },
)
