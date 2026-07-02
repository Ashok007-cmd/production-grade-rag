"""Package setup for Production-Grade RAG."""

from setuptools import find_packages, setup

setup(
    name="production-grade-rag",
    version="1.1.0",
    description="Production-Grade Retrieval-Augmented Generation pipeline",
    long_description=open("README.md").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    author="RAG Team",
    packages=find_packages(where=".", include=["src", "src.*"]),
    python_requires=">=3.11",
    install_requires=[
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
        "python-dotenv>=1.0.0",
        "pypdf>=3.0.0",
        "chromadb>=0.4.0",
        "sentence-transformers>=2.2.0",
        "rank-bm25>=0.2.0",
        "openai>=1.0.0",
    ],
    extras_require={
        "anthropic": ["anthropic>=0.30.0"],
        "dev": ["pytest>=7.4.0", "pytest-cov>=4.1.0", "tqdm>=4.65.0"],
        "all": ["anthropic>=0.30.0", "tqdm>=4.65.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
