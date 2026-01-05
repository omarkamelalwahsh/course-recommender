import pytest
import pandas as pd
import os
from src.recommender import CourseRecommender
from src.utils import normalize_query

@pytest.fixture
def recommender():
    rec = CourseRecommender()
    rec.load_courses("data/courses.csv")
    return rec

def test_strict_python_match(recommender):
    # Query: "Python" must return only courses mentioning Python
    res = recommender.recommend("Python", top_k=10)
    assert len(res["results"]) > 0
    for course in res["results"]:
        content = f"{course['title']} {course['skills']} {course['description']}".lower()
        assert "python" in content

def test_strict_flutter_match(recommender):
    # Query: "Flutter" must return only courses mentioning Flutter
    res = recommender.recommend("Flutter", top_k=5)
    # The dataset used in previous steps has No Flutter. 
    # If it's missing, it should return error.
    if not res["results"]:
        assert "❌ No courses found related to: flutter" in res["debug_info"]["error_message"]
    else:
        for course in res["results"]:
            content = f"{course['title']} {course['skills']} {course['description']}".lower()
            assert "flutter" in content

def test_missing_keyword_error(recommender):
    # Query: "Rust" (not in dataset) must return error message and no results
    res = recommender.recommend("Rust programming language", top_k=5)
    assert len(res["results"]) == 0
    assert "❌ No courses found related to: rust" in res["debug_info"]["error_message"]

def test_abbreviation_expansion(recommender):
    # Query: "ML" must behave as "machine learning"
    # We'll check if machine learning courses are found
    res = recommender.recommend("ML", top_k=5)
    # The cleaned query should be "machine learning"
    assert "machine learning" in res["debug_info"]["cleaned_query"]
    if res["results"]:
        for course in res["results"]:
            content = f"{course['title']} {course['skills']} {course['description']}".lower()
            assert "machine" in content or "learning" in content or "ml" in content

def test_rank_validity(recommender):
    # Rank must be integer between 1 and 10
    res = recommender.recommend("Javascript", top_k=10)
    if res["results"]:
        ranks = [c["rank"] for c in res["results"]]
        assert all(isinstance(r, int) for r in ranks)
        assert all(1 <= r <= 10 for r in ranks)
        if len(res["results"]) > 1:
            assert max(ranks) == 10
            assert min(ranks) == 1

def test_arabic_query(recommender):
    # Query: "انا عاوز اتعلم SEO" (I want to learn SEO)
    # Should ignore first 3 words, search for "SEO"
    res = recommender.recommend("انا عاوز اتعلم SEO", top_k=5)
    # "SEO" exists in dataset
    assert len(res["results"]) > 0
    assert "seo" in res["debug_info"]["cleaned_query"].lower()
    assert "انا" not in res["debug_info"]["cleaned_query"]
