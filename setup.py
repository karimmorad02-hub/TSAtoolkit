"""
aic_ts_suite – Modular Time-Series Notebook Toolkit
"""

from setuptools import setup, find_packages

setup(
    name="aic_ts_suite",
    version="0.1.0",
    description="Modular Time-Series Notebook Toolkit for R&D forecasting",
    author="Analytics Engineering Team",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "pandas>=2.0",
        "numpy>=1.24",
        "scipy>=1.10",
        "statsmodels>=0.14",
        "pmdarima>=2.0",
        "matplotlib>=3.7",
        "openpyxl>=3.1",
        "sqlalchemy>=2.0",
    ],
    extras_require={
        "db": ["psycopg2-binary>=2.9"],
        "ml": ["xgboost>=2.0", "prophet>=1.1"],
        "neural": ["neuralforecast>=1.6"],
        "timegpt": ["nixtla>=0.5"],
        "all": [
            "psycopg2-binary>=2.9",
            "xgboost>=2.0",
            "prophet>=1.1",
            "neuralforecast>=1.6",
            "nixtla>=0.5",
        ],
    },
)
