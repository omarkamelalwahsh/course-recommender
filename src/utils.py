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

# Stopwords lists
STOPWORDS_EN = {
    'i', 'want', 'to', 'learn', 'course', 'training', 'looking', 'interested', 'need', 
    'please', 'pls', 'can', 'could', 'would', 'should', 'the', 'a', 'an', 'and', 'or',
    'if', 'for', 'with', 'about', 'at', 'by', 'from', 'in', 'on'
}

STOPWORDS_AR = {
    'انا', 'عاوز', 'محتاج', 'اتعلم', 'كورس', 'شرح', 'عايز', 'من', 'فضلك', 'على', 'في',
    'الى', 'من', 'عن', 'يا', 'هل', 'كيف', 'ماذا'
}

def clean_text_basic(text: str) -> str:
    """Basic cleaning: lowercase and remove special characters."""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\u0600-\u06FF\s]', ' ', text)
    return " ".join(text.split())

def normalize_query(query: str, abbr_map: Dict[str, str]) -> str:
    """
    Normalize query: lowercase, remove punctuation, remove stopwords, expand abbreviations.
    """
    # 1. Lowercase and expand abbreviations with word boundaries
    q = query.lower()
    for abbr, full in sorted(abbr_map.items(), key=lambda x: len(x[0]), reverse=True):
        pattern = re.compile(r'\b' + re.escape(abbr.lower()) + r'\b')
        q = pattern.sub(f"{abbr} {full}", q)
    
    # 2. Remove punctuation and special characters
    q = re.sub(r'[^a-zA-Z0-9\u0600-\u06FF\s]', ' ', q)
    
    # 3. Remove stopwords
    tokens = q.split()
    cleaned_tokens = [t for t in tokens if t not in STOPWORDS_EN and t not in STOPWORDS_AR]
    
    return " ".join(cleaned_tokens).strip()

def infer_user_level(query: str) -> str:
    """
    Infer user level from query: White, Beginner, Intermediate, Advanced.
    """
    q = query.lower()
    
    # White Level Keywords
    white_k = ["from scratch", "zero knowledge", "مش فاهم", "من الصفر", "أبيض خالص", "مبتدئ جدا"]
    if any(k in q for k in white_k):
        return "White"
    
    # Keyword Dictionaries
    levels = {
        "Advanced": ["advanced", "expert", "deep dive", "professional", "senior", "master", "متقدم", "خبير", "باحتراف"],
        "Intermediate": ["intermediate", "medium", "moderate", "middle", "متوسط"],
        "Beginner": ["beginner", "basic", "intro", "introduction", "start", "مبتدئ", "أساسيات"]
    }

    scores = {lvl: 0 for lvl in levels}
    for lvl, keywords in levels.items():
        for k in keywords:
            if k in q:
                scores[lvl] += 1
        
    max_score = max(scores.values())
    if max_score > 0:
        for lvl in ["Advanced", "Intermediate", "Beginner"]:
            if scores[lvl] == max_score:
                return lvl
    
    return "Beginner"

def get_dataset_hash(df: pd.DataFrame) -> str:
    """Compute a stable hash of the dataframe content."""
    df_sorted = df.sort_index(axis=1).sort_index(axis=0)
    content = pd.util.hash_pandas_object(df_sorted, index=True).values
    return hashlib.md5(content).hexdigest()

def validate_and_clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and auto-fix schema."""
    df = df.copy()
    col_map = {'id':'course_id', 'name':'title', 'course_name':'title', 'tags':'skills', 'desc':'description'}
    df.rename(columns=lambda x: col_map.get(x.lower(), x), inplace=True)
    df.columns = [c.lower() for c in df.columns]

    if 'course_id' not in df.columns:
        df['course_id'] = list(range(1, len(df) + 1))
    
    required = {'title':'', 'category':'General', 'level':'Beginner', 'duration_hours':0.0, 'skills':'', 'description':'', 'instructor':'Unknown'}
    for col, default in required.items():
        if col not in df.columns: df[col] = default
        else: df[col] = df[col].fillna(default)

    link_cols = ["url", "link", "course_url", "course_link", "product_url"]
    df['course_link'] = ""
    for col in link_cols:
        if col in df.columns:
            df['course_link'] = df[col].fillna("")
            break

    def _norm_lvl(val):
        val = str(val).lower()
        if any(x in val for x in ['beg', 'white', 'intro']): return 'Beginner'
        if any(x in val for x in ['adv', 'exp']): return 'Advanced'
        return 'Intermediate'
    df['level'] = df['level'].apply(_norm_lvl)

    def _hours(val):
        try:
            if pd.isna(val) or val == '': return 0.0
            m = re.search(r"(\d+(\.\d+)?)", str(val))
            return float(m.group(1)) if m else 0.0
        except: return 0.0
    df['duration_hours'] = df['duration_hours'].apply(_hours)

    return df

def build_abbreviation_map(df: pd.DataFrame) -> Dict[str, str]:
    """Base abbreviations list."""
    return {
        'ml': 'machine learning',
        'nlp': 'natural language processing',
        'aws': 'amazon web services',
        'bi': 'business intelligence',
        'cv': 'computer vision',
        'ds': 'data science',
        'js': 'javascript',
        'sql': 'structured query language'
    }

def format_course_text(row: pd.Series, abbr_map: Dict[str, str] = None) -> str:
    """Format course info for search."""
    return f"{row['title']} {row['skills']} {row['description']}".lower()

def load_courses(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    return validate_and_clean_dataset(df)
