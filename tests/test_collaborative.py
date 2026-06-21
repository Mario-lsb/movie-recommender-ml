"""
test_collaborative.py
Unit tests for UserBasedRecommender and ItemBasedRecommender.
"""
import os
import sys
import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.collaborative.user_based import UserBasedRecommender # type: ignore
from src.collaborative.item_based import ItemBasedRecommender


@pytest.fixture
def sample_ratings():
    """Minimal ratings DataFrame for testing."""
    return pd.DataFrame({
        "user_id": [1, 1, 1, 2, 2, 3, 3, 3, 4, 4],
        "item_id": [1, 2, 3, 1, 4, 2, 3, 4, 1, 3],
        "rating":  [5, 3, 4, 4, 2, 5, 3, 4, 2, 5],
        "timestamp": [0] * 10
    })


# ── UserBasedRecommender tests ─────────────────────────────────────────────────

def test_user_based_fit_runs(sample_ratings):
    """fit() should run without errors."""
    model = UserBasedRecommender(k=2)
    model.fit(sample_ratings)


def test_user_based_matrix_shape(sample_ratings):
    """User-item matrix should have correct shape."""
    model = UserBasedRecommender(k=2)
    model.fit(sample_ratings)
    assert model.user_item_matrix.shape[0] == sample_ratings["user_id"].nunique()
    assert model.user_item_matrix.shape[1] == sample_ratings["item_id"].nunique()


def test_user_based_similarity_square(sample_ratings):
    """User similarity matrix should be square."""
    model = UserBasedRecommender(k=2)
    model.fit(sample_ratings)
    n = sample_ratings["user_id"].nunique()
    assert model.user_similarity.shape == (n, n)


def test_user_based_recommend_returns_series(sample_ratings):
    """recommend() should return a pandas Series."""
    model = UserBasedRecommender(k=2)
    model.fit(sample_ratings)
    result = model.recommend(user_id=1, n=3)
    assert isinstance(result, pd.Series)


def test_user_based_recommend_respects_n(sample_ratings):
    """recommend() should return at most n results."""
    model = UserBasedRecommender(k=2)
    model.fit(sample_ratings)
    result = model.recommend(user_id=1, n=2)
    assert len(result) <= 2


def test_user_based_scores_are_numeric(sample_ratings):
    """Recommendation scores should be numeric."""
    model = UserBasedRecommender(k=2)
    model.fit(sample_ratings)
    result = model.recommend(user_id=1, n=3)
    assert pd.api.types.is_numeric_dtype(result)


# ── ItemBasedRecommender tests ─────────────────────────────────────────────────

def test_item_based_fit_runs(sample_ratings):
    """fit() should run without errors."""
    model = ItemBasedRecommender(k=2)
    model.fit(sample_ratings)


def test_item_based_matrix_shape(sample_ratings):
    """User-item matrix should have correct shape."""
    model = ItemBasedRecommender(k=2)
    model.fit(sample_ratings)
    assert model.user_item_matrix.shape[0] == sample_ratings["user_id"].nunique() # type: ignore
    assert model.user_item_matrix.shape[1] == sample_ratings["item_id"].nunique() # type: ignore


def test_item_based_similarity_square(sample_ratings):
    """Item similarity matrix should be square."""
    model = ItemBasedRecommender(k=2)
    model.fit(sample_ratings)
    n = sample_ratings["item_id"].nunique()
    assert model.item_similarity.shape == (n, n) # type: ignore


def test_item_based_recommend_returns_series(sample_ratings):
    """recommend() should return a pandas Series."""
    model = ItemBasedRecommender(k=2)
    model.fit(sample_ratings)
    result = model.recommend(user_id=1, n=3)
    assert isinstance(result, pd.Series)


def test_item_based_recommend_respects_n(sample_ratings):
    """recommend() should return at most n results."""
    model = ItemBasedRecommender(k=2)
    model.fit(sample_ratings)
    result = model.recommend(user_id=1, n=2)
    assert len(result) <= 2


def test_item_based_scores_are_numeric(sample_ratings):
    """Recommendation scores should be numeric."""
    model = ItemBasedRecommender(k=2)
    model.fit(sample_ratings)
    result = model.recommend(user_id=1, n=3)
    assert pd.api.types.is_numeric_dtype(result)