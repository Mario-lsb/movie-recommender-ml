"""
test_content_based.py
Unit tests for ContentBasedRecommender.
"""
import os
import sys
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.content_based.tfidf_recommender import ContentBasedRecommender


@pytest.fixture
def sample_movies():
    """Minimal movies DataFrame for testing."""
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
def fitted_model(sample_movies):
    model = ContentBasedRecommender()
    model.fit(sample_movies)
    return model


# ── fit() tests ────────────────────────────────────────────────────────────────

def test_fit_runs_without_error(sample_movies):
    """fit() should complete without raising any exception."""
    model = ContentBasedRecommender()
    model.fit(sample_movies)  # should not raise


def test_fit_builds_tfidf_matrix(fitted_model, sample_movies):
    """TF-IDF matrix should have one row per movie."""
    assert fitted_model.tfidf_matrix is not None
    assert fitted_model.tfidf_matrix.shape[0] == len(sample_movies)


def test_fit_builds_cosine_similarity(fitted_model, sample_movies):
    """Cosine similarity matrix should be square (n_movies × n_movies)."""
    n = len(sample_movies)
    assert fitted_model.cosine_sim.shape == (n, n)


def test_fit_stores_movies_df(fitted_model, sample_movies):
    """fit() should store the movies DataFrame internally."""
    assert fitted_model.movies_df is not None
    assert len(fitted_model.movies_df) == len(sample_movies)


def test_fit_builds_indices(fitted_model, sample_movies):
    """fit() should build a title-to-index mapping."""
    assert fitted_model.indices is not None
    assert "Toy Story (1995)" in fitted_model.indices


# ── recommend() tests ──────────────────────────────────────────────────────────

def test_recommend_returns_dataframe(fitted_model):
    """recommend() should return a pandas DataFrame."""
    result = fitted_model.recommend("Toy Story (1995)", n=2)
    assert isinstance(result, pd.DataFrame)


def test_recommend_correct_columns(fitted_model):
    """Output DataFrame must contain title and similarity_score columns."""
    result = fitted_model.recommend("Toy Story (1995)", n=2)
    assert "title" in result.columns
    assert "similarity_score" in result.columns


def test_recommend_respects_n(fitted_model):
    """recommend() should return at most n results."""
    result = fitted_model.recommend("Toy Story (1995)", n=3)
    assert len(result) <= 3


def test_recommend_excludes_query_movie(fitted_model):
    """The queried movie itself must not appear in recommendations."""
    result = fitted_model.recommend("Toy Story (1995)", n=4)
    assert "Toy Story (1995)" not in result["title"].values


def test_recommend_scores_between_0_and_1(fitted_model):
    """Cosine similarity scores must be in [0, 1]."""
    result = fitted_model.recommend("Toy Story (1995)", n=4)
    assert (result["similarity_score"] >= 0).all()
    assert (result["similarity_score"] <= 1).all()


def test_recommend_unknown_title_raises(fitted_model):
    """recommend() should raise ValueError for an unknown movie title."""
    with pytest.raises(ValueError):
        fitted_model.recommend("Not A Real Movie (2099)", n=5)


def test_recommend_before_fit_raises():
    """Calling recommend() before fit() should raise an AttributeError or similar."""
    model = ContentBasedRecommender()
    with pytest.raises(Exception):
        model.recommend("Toy Story (1995)", n=5)