"""
Utility functions for the course recommender system.
"""
import json
import os
import re
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Tuple
import pandas as pd

# Strict Stopword Lists
STOPWORDS_EN = {
    'i', 'want', 'to', 'learn', 'course', 'courses', 'training', 'interested', 
    'need', 'looking', 'for', 'in', 'on', 'of', 'and', 'with', 'please', 'pls',
    'programming', 'language'
}

STOPWORDS_AR = {
    'انا', 'عاوز', 'عايز', 'محتاج', 'اتعلم', 'كورس', 'شرح', 'من', 'فضلك', 'في', 'على', 'عن'
}

def normalize_query(query: str) -> str:
    """
    Normalize query: lowercase, remove punctuation, remove extra spaces, remove stopwords.
    Expand abbreviations.
    """
    # 1. Lowercase
    q = query.lower()
    
    # 2. Expand Abbreviations
    abbr_map = {
        'ml': 'machine learning',
        'nlp': 'natural language processing',
        'aws': 'amazon web services',
        'bi': 'business intelligence',
        'cv': 'computer vision',
        'ds': 'data science'
    }
    for abbr, full in sorted(abbr_map.items(), key=lambda x: len(x[0]), reverse=True):
        pattern = re.compile(r'\b' + re.escape(abbr) + r'\b')
        q = pattern.sub(full, q)
        
    # 3. Remove Punctuation
    q = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', q)
    
    # 4. Remove Stopwords and extra spaces
    tokens = q.split()
    cleaned = [t for t in tokens if t not in STOPWORDS_EN and t not in STOPWORDS_AR]
    
    return " ".join(cleaned).strip()

def infer_user_level(query: str) -> str:
    """Infer user level from query."""
    q = query.lower()
    if any(k in q for k in ["from scratch", "zero knowledge", "مش فاهم", "من الصفر", "أبيض خالص"]):
        return "White"
    
    levels = {
        "Advanced": ["advanced", "expert", "deep dive", "professional", "senior", "master", "متقدم", "خبير"],
        "Intermediate": ["intermediate", "medium", "moderate", "middle", "متوسط"],
        "Beginner": ["beginner", "basic", "intro", "introduction", "start", "مبتدئ"]
    }
    scores = {lvl: 0 for lvl in levels}
    for lvl, keywords in levels.items():
        for k in keywords:
            if k in q: scores[lvl] += 1
    
    max_score = max(scores.values())
    if max_score > 0:
        for lvl in ["Advanced", "Intermediate", "Beginner"]:
            if scores[lvl] == max_score: return lvl
    return "Beginner"

def get_dataset_hash(df: pd.DataFrame) -> str:
    """Compute a stable hash of the dataframe content."""
    df_sorted = df.sort_index(axis=1).sort_index(axis=0)
    content = pd.util.hash_pandas_object(df_sorted, index=True).values
    return hashlib.md5(content).hexdigest()

def validate_and_clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize dataset columns and clean values."""
    df = df.copy()
    col_map = {'id':'course_id', 'name':'title', 'course_name':'title', 'tags':'skills', 'desc':'description'}
    df.rename(columns=lambda x: col_map.get(x.lower(), x), inplace=True)
    df.columns = [c.lower() for c in df.columns]

    if 'course_id' not in df.columns:
        df['course_id'] = list(range(1, len(df) + 1))
    
    # Required columns
    required = {
        'title': '', 'category': 'General', 'level': 'Beginner', 
        'duration_hours': 0.0, 'skills': '', 'description': '', 
        'instructor': 'Unknown', 'cover': ''
    }
    for col, default in required.items():
        if col not in df.columns: df[col] = default
        else: df[col] = df[col].fillna(default)

    # Detect link column
    link_cols = ["url", "link", "course_url", "course_link", "product_url"]
    df['course_link'] = ""
    for col in link_cols:
        if col in df.columns:
            df['course_link'] = df[col].fillna("")
            break

    # Normalize level
    def _norm(val):
        val = str(val).lower()
        if any(x in val for x in ['beg', 'white', 'intro']): return 'Beginner'
        if any(x in val for x in ['adv', 'exp']): return 'Advanced'
        return 'Intermediate'
    df['level'] = df['level'].apply(_norm)

    return df

def build_abbreviation_map(df: pd.DataFrame) -> Dict[str, str]:
    """Base abbreviations list."""
    return {
        'ml': 'machine learning', 'nlp': 'natural language processing',
        'aws': 'amazon web services', 'bi': 'business intelligence',
        'cv': 'computer vision', 'ds': 'data science'
    }

def format_course_text(row: pd.Series) -> str:
    """Format course info for search index."""
    return f"{row['title']} {row['skills']} {row['description']}".lower()

def load_courses(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset missing: {csv_path}")
    df = pd.read_csv(csv_path)
    return validate_and_clean_dataset(df)
