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
    Example: "ML" -> "ML Machine Learning"
    """
    query_lower = query.lower()
    expanded = query_lower
    
    # Sort by length descending to avoid partial replacements (e.g., 'NLP' vs 'NL')
    for abbr in sorted(abbr_map.keys(), key=len, reverse=True):
        full = abbr_map[abbr]
        # Match only whole words
        pattern = r'\b' + re.escape(abbr.lower()) + r'\b'
        if re.search(pattern, expanded):
            # Replace with "abbr full" to keep both for semantic matching
            expanded = re.sub(pattern, f"{abbr} {full}", expanded)
            
    return expanded


def infer_user_level(query: str) -> str:
    """
    Infer user level from query into: White, Beginner, Intermediate, Advanced.
    """
    query = query.lower()
    
    # White (Zero Knowledge) Keywords
    white_keywords = ["from scratch", "zero knowledge", "no experience", "start from zero", "مش فاهم حاجة", "من الصفر", "أبيض خالص", "مبتدئ جدا"]
    if any(k in query for k in white_keywords):
        return "White"
    
    # Advanced Keywords
    adv_keywords = ["advanced", "expert", "deep dive", "professional", "senior", "master", "متقدم", "خبير", "باحتراف", "عميق"]
    # Intermediate Keywords
    int_keywords = ["intermediate", "medium", "moderate", "middle", "متوسط", "مستوى متوسط"]
    # Beginner Keywords
    beg_keywords = ["beginner", "basic", "intro", "introduction", "start", "مبتدئ", "بداية", "أساسيات"]

    scores = {"Advanced": 0, "Intermediate": 0, "Beginner": 0}
    
    for k in adv_keywords:
        if k in query: scores["Advanced"] += 1
    for k in int_keywords:
        if k in query: scores["Intermediate"] += 1
    for k in beg_keywords:
        if k in query: scores["Beginner"] += 1
        
    # Return highest score if any, else None (to allow Pre-Level fallback)
    max_score = max(scores.values())
    if max_score > 0:
        return [k for k, v in scores.items() if v == max_score][0]
    
    return "Beginner" # Default fallback


def validate_and_clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and auto-fix the dataset schema for the Zedny dataset.
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

    # Required columns and their defaults
    if 'course_id' not in df.columns:
        df['course_id'] = list(range(1, len(df) + 1))
    
    defaults = {
        'title': '',
        'category': 'General',
        'level': 'Beginner',
        'duration_hours': 0.0,
        'skills': '',
        'description': '',
        'instructor': 'Unknown',
        'cover': ''
    }
    
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = df[col].fillna(default)
            if col == 'category':
                df[col] = df[col].replace(['', 'nan'], 'General')

    # Detect link column
    link_cols = ["url", "link", "course_url", "course_link", "product_url"]
    df['course_link'] = ""
    for col in link_cols:
        if col in df.columns:
            df['course_link'] = df[col].fillna("")
            break

    # Normalize 'level' strictly
    def normalize_level_internal(val):
        val = str(val).lower()
        if any(x in val for x in ['beg', 'jun', 'intro', 'start', 'white']):
            return 'Beginner' # "White" in dataset maps to Beginner for search
        if any(x in val for x in ['adv', 'exp', 'sen', 'deep', 'mast']):
            return 'Advanced'
        if any(x in val for x in ['inter', 'med', 'mid']):
            return 'Intermediate'
        return 'Intermediate'

    df['level'] = df['level'].apply(normalize_level_internal)

    # Clean 'duration_hours'
    def extract_hours(val):
        if pd.isna(val) or val == '': return 0.0
        if isinstance(val, (int, float)): return float(val)
        match = re.search(r"(\d+(\.\d+)?)", str(val))
        return float(match.group(1)) if match else 0.0

    df['duration_hours'] = df['duration_hours'].apply(extract_hours)

    # Auto Skill Extraction
    def extract_skills(row):
        skills = str(row['skills']).strip()
        if not skills or skills.lower() == 'nan':
            text = f"{row['title']} {row['description']}"
            tech_keywords = {
                'python', 'javascript', 'js', 'react', 'node', 'sql', 'html', 'css', 
                'java', 'c#', 'php', 'laravel', 'flutter', 'aws', 'azure', 'docker',
                'kubernetes', 'ml', 'ai', 'data science', 'marketing', 'sales',
                'excel', 'word', 'powerpoint', 'accounting', 'scrum', 'agile', 'wordpress'
            }
            extracted = set()
            words = re.findall(r'\b\w+\b', text.lower())
            for word in words:
                if word in tech_keywords:
                    extracted.add(word.capitalize())
            return "|".join(list(extracted)) if extracted else "General"
        return str(skills).replace(",", "|")

    df['skills'] = df.apply(extract_skills, axis=1)
    
    return df


def build_abbreviation_map(df: pd.DataFrame) -> Dict[str, str]:
    """
    Build abbreviation map: Global Tech + Auto-detected from dataset.
    Detects CAPS tokens (2-5 chars).
    """
    abbr_map = {
        'ml': 'machine learning',
        'dl': 'deep learning',
        'js': 'javascript',
        'nlp': 'natural language processing',
        'cv': 'computer vision',
        'ui/ux': 'user interface / user experience',
        'pm': 'project management',
        'bi': 'business intelligence',
        'aws': 'amazon web services',
        'sql': 'structured query language'
    }
    
    def clean_text(text):
        return re.sub(r'[^a-zA-Z0-9\s/]', '', str(text).lower()).strip()

    # Pattern for ALL CAPS (2-5 chars)
    abbr_pattern = re.compile(r'\b[A-Z0-9]{2,5}\b')
    # Pattern for "Full Form (ABBR)"
    extract_pattern = re.compile(r'\((?P<abbr>[A-Z0-9]{2,6})\)')
    
    for _, row in df.iterrows():
        text = f"{row['title']} {row['description']}"
        # 1. Look for explicit (ABBR)
        for m in extract_pattern.finditer(text):
            abbr = m.group('abbr').lower()
            span_end = m.start()
            pre_text = text[:span_end].strip()
            words = pre_text.split()
            if len(words) >= len(abbr):
                potential_full = " ".join(words[-len(abbr):])
                initials = "".join([w[0] for w in words[-len(abbr):] if w]).lower()
                if initials == abbr:
                   abbr_map[abbr] = clean_text(potential_full)
        
        # 2. Add candidates found in text if they are common (simplified: just add for now)
        # Note: In a real system we'd check frequency or initials nearby.
        
    return abbr_map


def format_course_text(row: pd.Series, abbr_map: Dict[str, str] = None) -> str:
    """
    Format course info for embedding.
    """
    text = f"{row['title']} {row['category']} {row['skills']} {row['description']}"
    return text.lower()


def load_courses(csv_path: str) -> pd.DataFrame:
    """
    Load data/courses.csv automatically.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset missing at {csv_path}")
    
    df = pd.read_csv(csv_path)
    return validate_and_clean_dataset(df)
