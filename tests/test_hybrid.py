"""
test_hybrid.py
Unit tests for HybridRecommender.
"""
import pytest  
import pandas as pd 
import numpy as np
import sys
import os
from unittest.mock import MagicMock

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.hybrid.hybrid_recommender import HybridRecommender
from src.content_based.tfidf_recommender import ContentBasedRecommender


@pytest.fixture
def sample_movies():
    return pd.DataFrame({
        "item_id":      [1, 2, 3, 4, 5],
        "title":        ["Toy Story (1995)", "GoldenEye (1995)", "Four Rooms (1995)",
                         "Get Shorty (1995)", "Copycat (1995)"],
        "release_date": ["01-Jan-1995"] * 5,
        "Action":       [0, 1, 0, 0, 0],
        "Adventure":    [0, 1, 0, 0, 0],
        "Animation":    [1, 0, 0, 0, 0],
        "Childrens":    [1, 0, 0, 0, 0],
        "Comedy":       [1, 0, 1, 1, 0],
        "Crime":        [0, 0, 0, 1, 0],
        "Documentary":  [0, 0, 0, 0, 0],
        "Drama":        [0, 0, 1, 0, 1],
        "Fantasy":      [0, 0, 0, 0, 0],
        "Film_Noir":    [0, 0, 0, 0, 0],
        "Horror":       [0, 0, 0, 0, 1],
        "Musical":      [0, 0, 0, 0, 0],
        "Mystery":      [0, 0, 0, 0, 1],
        "Romance":      [0, 0, 0, 0, 0],
        "Sci_Fi":       [0, 0, 0, 0, 0],
        "Thriller":     [0, 0, 1, 0, 1],
        "War":          [0, 0, 0, 0, 0],
        "Western":      [0, 0, 0, 0, 0],
    })


@pytest.fixture
def mock_cf_model():
    """Mock Surprise SVD model that returns a fake prediction."""
    mock = MagicMock()
    mock.predict.return_value = MagicMock(est=3.5)
    return mock


@pytest.fixture
def fitted_content_model(sample_movies):
    model = ContentBasedRecommender()
    model.fit(sample_movies)
    return model


@pytest.fixture
def hybrid(mock_cf_model, fitted_content_model):
    return HybridRecommender(
        cf_model=mock_cf_model,
        content_model=fitted_content_model,
        alpha=0.7
    )


# ── Initialization tests ───────────────────────────────────────────────────────

def test_hybrid_default_alpha(mock_cf_model, fitted_content_model):
    """Default alpha should be 0.7."""
    h = HybridRecommender(cf_model=mock_cf_model, content_model=fitted_content_model)
    assert h.alpha == 0.7


def test_hybrid_custom_alpha(mock_cf_model, fitted_content_model):
    """Alpha should be settable at construction and by assignment."""
    h = HybridRecommender(mock_cf_model, fitted_content_model, alpha=0.3)
    assert h.alpha == 0.3
    h.alpha = 0.5
    assert h.alpha == 0.5


# ── recommend() tests ──────────────────────────────────────────────────────────

def test_recommend_returns_dataframe(hybrid, sample_movies):
    result = hybrid.recommend(user_id=1, movies_df=sample_movies, n=3)
    assert isinstance(result, pd.DataFrame)


def test_recommend_correct_columns(hybrid, sample_movies):
    result = hybrid.recommend(user_id=1, movies_df=sample_movies, n=3)
    assert "title" in result.columns
    assert "hybrid_score" in result.columns
    assert "item_id" in result.columns


def test_recommend_respects_n(hybrid, sample_movies):
    result = hybrid.recommend(user_id=1, movies_df=sample_movies, n=3)
    assert len(result) <= 3


def test_recommend_scores_are_numeric(hybrid, sample_movies):
    result = hybrid.recommend(user_id=1, movies_df=sample_movies, n=5)
    assert pd.api.types.is_numeric_dtype(result["hybrid_score"])


def test_recommend_scores_between_0_and_1(hybrid, sample_movies):
    """Hybrid scores are weighted averages of normalized values — must be in [0,1]."""
    result = hybrid.recommend(user_id=1, movies_df=sample_movies, n=5)
    assert (result["hybrid_score"] >= 0).all()
    assert (result["hybrid_score"] <= 1).all()


def test_recommend_sorted_descending(hybrid, sample_movies):
    """Results should be sorted highest score first."""
    result = hybrid.recommend(user_id=1, movies_df=sample_movies, n=5)
    scores = result["hybrid_score"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_recommend_alpha_zero_uses_content_only(mock_cf_model, fitted_content_model, sample_movies):
    """With alpha=0, CF scores are ignored — only content scores matter."""
    h = HybridRecommender(mock_cf_model, fitted_content_model, alpha=0.0)
    result = h.recommend(user_id=1, movies_df=sample_movies, n=5)
    assert len(result) > 0


def test_recommend_alpha_one_uses_cf_only(mock_cf_model, fitted_content_model, sample_movies):
    """With alpha=1, content scores are ignored — only CF scores matter."""
    h = HybridRecommender(mock_cf_model, fitted_content_model, alpha=1.0)
    result = h.recommend(user_id=1, movies_df=sample_movies, n=5)
    assert len(result) > 0


def test_recommend_calls_cf_for_every_item(mock_cf_model, fitted_content_model, sample_movies):
    """CF model's predict() should be called once per movie."""
    h = HybridRecommender(mock_cf_model, fitted_content_model, alpha=0.7)
    h.recommend(user_id=1, movies_df=sample_movies, n=3)
    assert mock_cf_model.predict.call_count == len(sample_movies)