"""
Setup script for the Trading Advisor package.
"""

from setuptools import setup, find_packages

setup(
    name="tradingagents",
    version="0.1.0",
    description="Trading Advisor - AI-Powered Multi-Agent Trading Advisory System",
    author="jerryzhang1011",
    author_email="",
    url="https://github.com/jerryzhang1011/tradingAdvisor",
    packages=find_packages(),
    install_requires=[
        "langchain>=0.1.0",
        "langchain-openai>=0.0.2",
        "langchain-experimental>=0.0.40",
        "langgraph>=0.0.20",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "praw>=7.7.0",
        "stockstats>=0.5.4",
        "yfinance>=0.2.31",
        "typer>=0.9.0",
        "rich>=13.0.0",
        "questionary>=2.0.1",
    ],
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "tradingadvisor=cli.main:app",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Trading Industry",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
)
