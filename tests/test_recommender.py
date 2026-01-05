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

def test_abbreviation_expansion():
    abbr_map = {"ML": "Machine Learning", "NLP": "Natural Language Processing"}
    # Word boundary test
    assert "ML Machine Learning" in normalize_query("I want to learn ML", abbr_map)
    # Case insensitive check
    assert "ML Machine Learning" in normalize_query("i like ml", abbr_map)
    # Non-word boundary check (should NOT expand)
    assert "html" == normalize_query("html", abbr_map).strip()

def test_level_inference():
    # English
    assert infer_user_level("learn python from scratch") == "White"
    assert infer_user_level("advanced expert deep dive") == "Advanced"
    
    # Arabic
    assert infer_user_level("من الصفر مش فاهم أي حاجة") == "White"
    assert infer_user_level("باحتراف خبير متقدم") == "Advanced"
    assert infer_user_level("مستوى متوسط") == "Intermediate"
    assert infer_user_level("بداية أساسيات") == "Beginner"

def test_rank_normalization(recommender):
    res = recommender.recommend("python", top_k=5)
    results = res["results"]
    if results:
        ranks = [r["rank"] for r in results]
        assert all(isinstance(r, int) for r in ranks)
        assert all(1 <= r <= 10 for r in ranks)
        if len(results) > 1:
            assert max(ranks) == 10
            assert min(ranks) == 1

def test_keyword_guardrail(recommender):
    # Search for non-existent tech
    res = recommender.recommend("flutter", top_k=5)
    assert len(res["results"]) == 0
    assert "No courses found related to: flutter" in res["debug_info"]["keyword_warning"]

def test_filler_words(recommender):
    # Filler words should NOT trigger warning if core exists
    # "want learn python" -> "python" exists
    res = recommender.recommend("I want to learn python", top_k=5)
    assert len(res["results"]) > 0
    assert res["debug_info"]["keyword_warning"] is None
