from setuptools import setup, find_packages

setup(
    name="CineMatch-ML",
    version="1.0.0",
    description="ML-powered movie recommendation engine using SVD + TF-IDF hybrid model",
    author="MEGHANA_KAMATAM",
    author_email="your-email@gmail.com",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "scikit-surprise",
        "matplotlib",
        "seaborn",
        "streamlit",
        "joblib",
        "jupyter",
    ],
)