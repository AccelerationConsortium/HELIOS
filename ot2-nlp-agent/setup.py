"""Setup script for OT-2 NLP Agent."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="ot2-nlp-agent",
    version="0.1.0",
    author="Battery Lab",
    description="Natural language interface for Opentrons OT-2 robot",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/ot2-nlp-agent",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.8",
    install_requires=[
        # No required dependencies for basic functionality
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "ruff>=0.1.0",
            "mypy>=1.0.0",
        ],
        "llm": [
            "openai>=1.0.0",
            "anthropic>=0.18.0",
        ],
        "ui": [
            "gradio>=4.0.0",
            "fastapi>=0.100.0",
            "uvicorn>=0.20.0",
        ],
    },
)
