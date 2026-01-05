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


def save_recommendations(
    user_query: str,
    filters: Dict[str, Any],
    recommendations: List[Dict[str, Any]],
    output_path: str = "outputs/recommendations.json"
) -> None:
    """
    Save recommendation results to JSON file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    result = {
        "timestamp": datetime.now().isoformat(),
        "user_query": user_query,
        "filters": filters,
        "recommended_courses": recommendations
    }
    
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                if not isinstance(data, list):
                    data = [data]
            except json.JSONDecodeError:
                data = []
    else:
        data = []
    
    data.append(result)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_dataset_hash(df: pd.DataFrame) -> str:
    """
    Compute a stable hash of the dataframe content.
    """
    df_sorted = df.sort_index(axis=1).sort_index(axis=0)
    content = pd.util.hash_pandas_object(df_sorted, index=True).values
    return hashlib.md5(content).hexdigest()


def normalize_query(query: str, abbr_map: Dict[str, str]) -> str:
    """
    Expand abbreviations in user query using word boundaries.
    Apply only to whole words.
    """
    expanded = query
    # Sort by length descending to avoid partial replacements
    for abbr in sorted(abbr_map.keys(), key=len, reverse=True):
        full = abbr_map[abbr]
        # Regex for word boundaries, case-insensitive
        pattern = re.compile(r'\b' + re.escape(abbr) + r'\b', re.IGNORECASE)
        # We replace "ML" with "ML Machine Learning" to help both keyword and semantic search
        expanded = pattern.sub(f"{abbr} {full}", expanded)
            
    return expanded


def infer_user_level(query: str) -> str:
    """
    Infer user level from query: White, Beginner, Intermediate, Advanced.
    """
    q = query.lower()
    
    # White Level Keywords
    white_k = ["from scratch", "zero knowledge", "مش فاهم", "من الصفر", "أبيض خالص", "مبتدئ جدا", "ابدا من الاول"]
    if any(k in q for k in white_k):
        return "White"
    
    # Keyword Dictionaries
    levels = {
        "Advanced": ["advanced", "expert", "deep dive", "professional", "senior", "master", "متقدم", "خبير", "باحتراف", "عميق"],
        "Intermediate": ["intermediate", "medium", "moderate", "middle", "متوسط", "مستوى متوسط"],
        "Beginner": ["beginner", "basic", "intro", "introduction", "start", "مبتدئ", "بداية", "أساسيات"]
    }

    scores = {lvl: 0 for lvl in levels}
    for lvl, keywords in levels.items():
        for k in keywords:
            if k in q:
                scores[lvl] += 1
        
    max_score = max(scores.values())
    if max_score > 0:
        # Resolve ties by picking highest level
        for lvl in ["Advanced", "Intermediate", "Beginner"]:
            if scores[lvl] == max_score:
                return lvl
    
    return "Beginner" # Default


def validate_and_clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and auto-fix schema.
    """
    df = df.copy()
    
    # Standardize column names
    col_map = {
        'id': 'course_id',
        'name': 'title',
        'course_name': 'title',
        'tags': 'skills',
        'desc': 'description'
    }
    df.rename(columns=lambda x: col_map.get(x.lower(), x), inplace=True)
    df.columns = [c.lower() for c in df.columns]

    # Defaults
    if 'course_id' not in df.columns:
        df['course_id'] = list(range(1, len(df) + 1))
    
    required = {
        'title': '',
        'category': 'General',
        'level': 'Beginner',
        'duration_hours': 0.0,
        'skills': '',
        'description': '',
        'instructor': 'Unknown'
    }
    
    for col, default in required.items():
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = df[col].fillna(default)

    # DETECT LINK COLUMN
    link_cols = ["url", "link", "course_url", "course_link", "product_url"]
    df['course_link'] = ""
    for col in link_cols:
        if col in df.columns:
            df['course_link'] = df[col].fillna("")
            break

    # Normalize level
    def _norm_lvl(val):
        val = str(val).lower()
        if any(x in val for x in ['beg', 'white', 'intro']): return 'Beginner'
        if any(x in val for x in ['adv', 'exp', 'prof']): return 'Advanced'
        return 'Intermediate'

    df['level'] = df['level'].apply(_norm_lvl)

    # Duration Hours
    def _hours(val):
        try:
            if pd.isna(val) or val == '': return 0.0
            if isinstance(val, (int, float)): return float(val)
            m = re.search(r"(\d+(\.\d+)?)", str(val))
            return float(m.group(1)) if m else 0.0
        except: return 0.0
    df['duration_hours'] = df['duration_hours'].apply(_hours)

    return df


def build_abbreviation_map(df: pd.DataFrame) -> Dict[str, str]:
    """
    Global + Dataset-specific abbreviations.
    """
    abbr_map = {
        'ML': 'Machine Learning',
        'NLP': 'Natural Language Processing',
        'DL': 'Deep Learning',
        'CV': 'Computer Vision',
        'BI': 'Business Intelligence',
        'AWS': 'Amazon Web Services',
        'SQL': 'Structured Query Language',
        'JS': 'JavaScript'
    }
    
    # Detect caps tokens (2-5 chars) - Simplified for this task
    # We could scan df text, but global list covers the target requirement.
    return abbr_map


def format_course_text(row: pd.Series, abbr_map: Dict[str, str] = None) -> str:
    """
    Format course info for embedding.
    """
    return f"{row['title']} {row['skills']} {row['description']}".lower()


def load_courses(csv_path: str) -> pd.DataFrame:
    """
    Auto-load dataset.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    return validate_and_clean_dataset(df)
