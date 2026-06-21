"""
tfidf_recommender.py
Content-based filtering using TF-IDF on movie genres and metadata.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    def __init__(self):
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.movies_df = None
        self.indices = None

    def _build_genre_string(self, row):
        """Combine genre columns into a single string for TF-IDF."""
        genre_cols = [
            "Action", "Adventure", "Animation", "Childrens", "Comedy",
            "Crime", "Documentary", "Drama", "Fantasy", "Film_Noir",
            "Horror", "Musical", "Mystery", "Romance", "Sci_Fi",
            "Thriller", "War", "Western",
        ]
        genres = [col for col in genre_cols if row.get(col, 0) == 1]
        return " ".join(genres)

    def fit(self, movies_df: pd.DataFrame):
        """
        Build TF-IDF matrix from movie metadata.

        Args:
            movies_df: DataFrame with movie metadata and genre columns
        """
        self.movies_df = movies_df.copy().reset_index(drop=True)
        self.movies_df["genres_str"] = self.movies_df.apply(self._build_genre_string, axis=1)

        tfidf = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = tfidf.fit_transform(self.movies_df["genres_str"])
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        self.indices = pd.Series(self.movies_df.index, index=self.movies_df["title"])

    def recommend(self, title: str, n: int = 10):
        """
        Get top-N similar movies by content similarity.

        Args:
            title: exact movie title from dataset
            n: number of recommendations

        Returns:
            DataFrame of top-N recommended movies with similarity scores
        """
        if title not in self.indices: # type: ignore
            raise ValueError(f"Movie '{title}' not found in dataset.")

        idx = self.indices[title] # type: ignore
        sim_scores = list(enumerate(self.cosine_sim[idx])) # type: ignore
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1: n + 1]  # exclude the movie itself

        movie_indices = [i[0] for i in sim_scores]
        result = self.movies_df.iloc[movie_indices][["title", "release_date"]].copy() # type: ignore
        result["similarity_score"] = [round(i[1], 4) for i in sim_scores]
        return result.reset_index(drop=True)
