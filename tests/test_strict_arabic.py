
import pytest
from src.pipeline import CourseRecommenderPipeline
from src.schemas import RecommendRequest

@pytest.fixture(scope="module")
def pipeline():
    return CourseRecommenderPipeline()

def test_strict_php_arabic(pipeline):
    """Test: 'بي اتش بي' -> PHP courses."""
    # Our map converts 'بي اتش بي' -> 'php'
    req = RecommendRequest(query="بي اتش بي", top_k=5)
    res = pipeline.recommend(req)
    
    assert res.total_found > 0, "Should find PHP courses"
    for r in res.results:
        print(f"PHP Result: {r.title}")
        # 'php' is strict. Matches must contain it.
        # Description of PHP course (Line 16) mentions PHP explicitly.
        is_php = 'php' in r.title.lower() or 'php' in r.matched_keywords
        assert is_php

def test_strict_c_sharp_arabic(pipeline):
    """Test: 'سي شارب' -> C# courses."""
    # Map: 'سي شارب' -> 'c#'
    req = RecommendRequest(query="سي شارب", top_k=5)
    res = pipeline.recommend(req)
    
    assert res.total_found > 0, "Should find C# courses"
    for r in res.results:
        print(f"C# Result: {r.title}")
        # Keyword 'c#' must be matched.
        # Note: Title might be 'C 2019...' but description contains 'C#'.
        assert 'c#' in r.matched_keywords

def test_strict_java_returns_nothing(pipeline):
    """
    Test: 'جافا' (Java).
    Dataset HAS NO Java courses (only JavaScript).
    Strict logic checks 'java' keyword.
    'java' should NOT match 'javascript' due to \b boundary.
    Therefore, should return 0 results.
    """
    req = RecommendRequest(query="جافا", top_k=5)
    res = pipeline.recommend(req)
    
    # If this is 0, it PROVES strictness! 
    # Because a "Loose" sematic search would definitely return JavaScript (high similarity).
    assert res.total_found == 0, f"Found {res.total_found} results for Java (expected 0)"

def test_strict_marketing_arabic(pipeline):
    """Test: 'تسويق' -> Marketing."""
    req = RecommendRequest(query="تسويق", top_k=5)
    res = pipeline.recommend(req)
    
    assert res.total_found > 0
    for r in res.results:
        print(f"Mkt Result: {r.title}")
        assert 'marketing' in r.matched_keywords or 'marketing' in r.title.lower()

def test_irrelevant_returns_nothing(pipeline):
    """Test: 'طبخ' (Cooking) -> Nothing."""
    req = RecommendRequest(query="طبخ", top_k=5)
    res = pipeline.recommend(req)
    assert res.total_found == 0

def test_typo_javascript_space(pipeline):
    """Test: 'java script' -> Should find JavaScript courses, NOT Java/Android."""
    req = RecommendRequest(query="java script", top_k=5)
    res = pipeline.recommend(req)
    
    assert res.total_found > 0
    for r in res.results:
        # Should be JS related
        print(f"JS Typo Result: {r.title}")
        is_js = 'javascript' in r.title.lower() or 'javascript' in r.matched_keywords
        assert is_js, f"Expected JavaScript course, got {r.title}"
