"""
preprocessor.py
Data preprocessing utilities.
"""
import pandas as pd
import numpy as np


def handle_missing_values(movies_df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing genre columns with 0."""
    genre_cols = [
        "Action", "Adventure", "Animation", "Childrens", "Comedy",
        "Crime", "Documentary", "Drama", "Fantasy", "Film_Noir",
        "Horror", "Musical", "Mystery", "Romance", "Sci_Fi",
        "Thriller", "War", "Western",
    ]
    for col in genre_cols:
        if col in movies_df.columns:
            movies_df[col] = movies_df[col].fillna(0).astype(int)
    return movies_df


def normalize_ratings(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize ratings to [0, 1] range."""
    ratings_df = ratings_df.copy()
    min_r = ratings_df["rating"].min()
    max_r = ratings_df["rating"].max()
    ratings_df["rating_normalized"] = (ratings_df["rating"] - min_r) / (max_r - min_r)
    return ratings_df


def clean_movies(movies_df: pd.DataFrame) -> pd.DataFrame:
    """Clean movies DataFrame."""
    movies_df = movies_df.copy()
    movies_df = handle_missing_values(movies_df)
    movies_df = movies_df.dropna(subset=["title"])
    movies_df = movies_df.reset_index(drop=True)
    return movies_df


def clean_ratings(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """Clean ratings DataFrame."""
    ratings_df = ratings_df.copy()
    ratings_df = ratings_df.dropna()
    ratings_df = ratings_df[ratings_df["rating"].between(1, 5)]
    ratings_df = ratings_df.reset_index(drop=True)
    return ratings_df