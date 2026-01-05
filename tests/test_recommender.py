import pytest
import pandas as pd
import os
from src.recommender import CourseRecommender
from src.utils import validate_and_clean_dataset, normalize_query, infer_user_level

@pytest.fixture
def recommender():
    rec = CourseRecommender()
    rec.load_courses("data/courses.csv")
    return rec

def test_dataset_loading(recommender):
    assert recommender.courses_df is not None
    assert not recommender.courses_df.empty
    expected_cols = {'course_id', 'title', 'category', 'level', 'duration_hours', 'skills', 'description', 'instructor', 'cover', 'course_link'}
    assert expected_cols.issubset(set(recommender.courses_df.columns))

def test_level_inference():
    # English
    assert infer_user_level("learn python from scratch") == "White"
    assert infer_user_level("advanced machine learning expert") == "Advanced"
    assert infer_user_level("intermediate javascript") == "Intermediate"
    assert infer_user_level("beginner java") == "Beginner"
    
    # Arabic
    assert infer_user_level("كورس من الصفر") == "White"
    assert infer_user_level("شرح متقدم جدا") == "Advanced"
    assert infer_user_level("مستوى متوسط") == "Intermediate"
    assert infer_user_level("أساسيات البرمجة") == "Beginner"

def test_abbreviation_expansion():
    abbr_map = {"ml": "machine learning", "js": "javascript"}
    
    # Word boundary check
    assert "ml machine learning" in normalize_query("I want to learn ML", abbr_map)
    # Ensure it doesn't expand parts of words
    assert "html" == normalize_query("html", abbr_map).strip()

def test_ranking_logic(recommender):
    response = recommender.recommend("python", top_k=10)
    results = response["results"]
    if results:
        ranks = [r["rank"] for r in results]
        assert all(isinstance(r, int) for r in ranks)
        assert all(1 <= r <= 10 for r in ranks)
        assert max(ranks) == 10
        if len(results) > 1:
            assert min(ranks) == 1

def test_keyword_guardrail(recommender):
    # Search for something that doesn't exist
    response = recommender.recommend("flutter", top_k=5)
    assert len(response["results"]) == 0
    assert "keyword_warning" in response["debug_info"]
    assert "No courses found related to: flutter" in response["debug_info"]["keyword_warning"]

def test_stopword_filtering(recommender):
    # Query with filler words shouldn't trigger guardrail if the core keyword exists
    # "want learn javascript" -> "javascript" is in data
    response = recommender.recommend("I want to learn javascript", top_k=5)
    assert len(response["results"]) > 0
    assert response["debug_info"]["keyword_warning"] is None
