import pytest
import pandas as pd
import os
from src.recommender import CourseRecommender
from src.utils import normalize_query, infer_user_level

@pytest.fixture
def recommender():
    rec = CourseRecommender()
    rec.load_courses("data/courses.csv")
    return rec

def test_query_normalization():
    abbr_map = {"ml": "machine learning", "ds": "data science"}
    # Lowercase, punctuation, and expansion
    q = "I want to learn ML!"
    norm = normalize_query(q, abbr_map)
    assert "machine learning" in norm
    assert "!" not in norm
    assert "want" not in norm # Stopword

def test_relevance_guardrail(recommender):
    # Search for technology that DEFINITELY isn't in the dataset
    # The requirement is that it should return NO RESULTS if core keyword is missing
    res = recommender.recommend("Flutter development", top_k=5)
    assert len(res["results"]) == 0
    assert "No courses found related to: flutter" in res["debug_info"]["keyword_warning"]

def test_positive_match(recommender):
    # Valid query that should return results
    res = recommender.recommend("Python programming", top_k=5)
    assert len(res["results"]) > 0
    # Check if keyword score is computed
    assert res["results"][0]['keyword_score'] > 0

def test_abbreviation_expansion(recommender):
    # JS should expand and match
    res = recommender.recommend("Advanced JS", top_k=5)
    assert len(res["results"]) > 0
    # The cleaned query should contain javascript
    assert "javascript" in res["debug_info"]["cleaned_query"]

def test_rank_integrity(recommender):
    res = recommender.recommend("data science", top_k=10)
    results = res["results"]
    if results:
        ranks = [r["rank"] for r in results]
        assert all(isinstance(r, int) for r in ranks)
        assert all(1 <= r <= 10 for r in ranks)
        if len(results) > 1:
            assert max(ranks) == 10
            assert min(ranks) == 1

def test_level_inference():
    assert infer_user_level("I am a beginner from scratch") == "White"
    assert infer_user_level("advanced expert masterclass") == "Advanced"
    assert infer_user_level("intermediate level data science") == "Intermediate"
