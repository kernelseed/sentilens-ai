"""
SentilensAI - Setup Configuration

Setup script for SentilensAI sentiment analysis package.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="sentilens-ai",
    version="1.0.0",
    author="Pravin Selvamuthu",
    author_email="pravin.selvamuthu@gmail.com",
    description="Advanced sentiment analysis for AI chatbot messages using LangChain and machine learning",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/kernelseed/sentilens-ai",
    project_urls={
        "Bug Tracker": "https://github.com/kernelseed/sentilens-ai/issues",
        "Documentation": "https://github.com/kernelseed/sentilens-ai/wiki",
        "Source Code": "https://github.com/kernelseed/sentilens-ai",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-cov>=4.1.0",
            "black>=23.12.1",
            "flake8>=6.1.0",
            "mypy>=1.8.0",
            "pre-commit>=3.6.0",
        ],
        "visualization": [
            "matplotlib>=3.9.2",
            "seaborn>=0.13.2",
            "plotly>=5.17.0",
            "wordcloud>=1.9.2",
        ],
        "api": [
            "fastapi>=0.108.0",
            "uvicorn>=0.25.0",
            "pydantic>=2.5.2",
        ],
        "advanced-ml": [
            "xgboost>=2.1.3",
            "lightgbm>=4.1.0",
            "catboost>=1.2.2",
        ],
    },
    entry_points={
        "console_scripts": [
        "sentilens-ai-analyze=sentiment_analyzer:main",
        "sentilens-ai-train=ml_training_pipeline:main",
        "sentilens-ai-integrate=chatbot_integration:main",
        "sentilens-ai-visualize=visualization:main",
        ],
    },
    include_package_data=True,
    package_data={
        "sentilens_ai": [
            "*.json",
            "*.yaml",
            "*.yml",
            "templates/*",
            "static/*",
        ],
    },
    keywords=[
        "sentiment-analysis",
        "chatbot",
        "ai",
        "machine-learning",
        "langchain",
        "nlp",
        "natural-language-processing",
        "emotion-detection",
        "conversation-analysis",
        "artificial-intelligence",
    ],
    zip_safe=False,
)
