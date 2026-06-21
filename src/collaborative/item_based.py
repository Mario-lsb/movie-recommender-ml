"""
item_based.py
Item-based collaborative filtering.
"""
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class ItemBasedRecommender:
    def __init__(self, k=20):
        self.k = k
        self.user_item_matrix = None
        self.item_similarity = None

    def fit(self, ratings_df: pd.DataFrame):
        self.user_item_matrix = ratings_df.pivot_table(
            index='user_id', columns='item_id', values='rating'
        ).fillna(0)
        item_matrix = self.user_item_matrix.T
        self.item_similarity = cosine_similarity(item_matrix)
        self.item_similarity = pd.DataFrame(
            self.item_similarity,
            index=item_matrix.index,
            columns=item_matrix.index
        )

    def recommend(self, user_id: int, n: int = 10):
        user_ratings = self.user_item_matrix.loc[user_id] # type: ignore
        rated_items = user_ratings[user_ratings > 0]
        unrated_items = user_ratings[user_ratings == 0].index
        scores = {}
        for item_id in unrated_items:
            if item_id not in self.item_similarity:
                continue
            sim_scores = self.item_similarity[item_id][rated_items.index] # type: ignore
            scores[item_id] = np.dot(sim_scores, rated_items) / (sim_scores.sum() + 1e-8)
        return pd.Series(scores).nlargest(n)