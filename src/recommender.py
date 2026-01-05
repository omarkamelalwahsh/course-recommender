import os
import json
import pandas as pd
import numpy as np
import re
from typing import List, Dict, Any, Optional, Union
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    SentenceTransformer = None
    cosine_similarity = None

from src.utils import (
    load_courses, 
    format_course_text, 
    validate_and_clean_dataset,
    build_abbreviation_map,
    get_dataset_hash,
    normalize_query,
    infer_user_level
)

class CourseRecommender:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', model: Any = None):
        """
        Initialize the Course Recommender system.
        """
        self.model_name = model_name
        self.model = model
        self.courses_df = None
        self.embeddings = None
        self.abbr_map = {}
        self.dataset_hash = None
        
        # Fallback data
        self.fallback_data = [
            {"course_id": 1, "title": "Python for Beginners", "category": "Programming", "level": "Beginner", "duration_hours": 10.0, "skills": "Python", "description": "Basics", "instructor": "Unknown", "course_link": ""},
            {"course_id": 2, "title": "Advanced ML", "category": "Data Science", "level": "Advanced", "duration_hours": 20.0, "skills": "ML", "description": "Expert", "instructor": "Unknown", "course_link": ""}
        ]

    def _initialize_model(self):
        """Initialize the sentence transformer model."""
        if self.model is None and SentenceTransformer is not None:
            print(f"Loading model: {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
            print("Model loaded.")

    def load_courses(self, source: Union[str, pd.DataFrame]) -> None:
        """
        Load courses from CSV path or DataFrame with caching.
        """
        try:
            if isinstance(source, pd.DataFrame):
                self.courses_df = validate_and_clean_dataset(source)
            else:
                if os.path.exists(source):
                    raw_df = pd.read_csv(source)
                    self.courses_df = validate_and_clean_dataset(raw_df)
                else:
                    raise FileNotFoundError(f"File not found: {source}")
            
            self.dataset_hash = get_dataset_hash(self.courses_df)
            
            cache_dir = "outputs"
            os.makedirs(cache_dir, exist_ok=True)
            
            emb_path = os.path.join(cache_dir, f"embeddings_{self.dataset_hash}.npy")
            map_path = os.path.join(cache_dir, f"abbr_map_{self.dataset_hash}.json")
            
            if os.path.exists(emb_path) and os.path.exists(map_path):
                print("Loading from cache...")
                self.embeddings = np.load(emb_path)
                with open(map_path, 'r') as f:
                    self.abbr_map = json.load(f)
                
                self.courses_df['combined_text'] = self.courses_df.apply(
                    lambda row: format_course_text(row, self.abbr_map), axis=1
                )
                self._initialize_model()
            else:
                print("No cache found. Computing...")
                self.abbr_map = build_abbreviation_map(self.courses_df)
                self._compute_embeddings()
                
                if self.embeddings is not None:
                    np.save(emb_path, self.embeddings)
                with open(map_path, 'w') as f:
                    json.dump(self.abbr_map, f)
                    
        except Exception as e:
            print(f"Error loading: {e}. Using fallback.")
            self.courses_df = validate_and_clean_dataset(pd.DataFrame(self.fallback_data))
            self.abbr_map = build_abbreviation_map(self.courses_df)
            self._compute_embeddings()

    def _compute_embeddings(self) -> None:
        """Compute embeddings for all courses."""
        if self.courses_df is None or self.courses_df.empty:
            return

        self.courses_df['combined_text'] = self.courses_df.apply(
            lambda row: format_course_text(row, self.abbr_map), axis=1
        )
        
        self._initialize_model()
        
        if self.model:
            print("Computing embeddings...")
            self.embeddings = self.model.encode(self.courses_df['combined_text'].tolist(), show_progress_bar=True)
            print("Embeddings computed.")

    def recommend(
        self, 
        user_query: str, 
        top_k: int = 30, 
        pre_filters: Optional[Dict[str, Any]] = None,
        similarity_threshold: float = 0.2
    ) -> Dict[str, Any]:
        """
        Get course recommendations with query expansion, level inference, and strict 1–10 ranking.
        """
        if self.courses_df is None:
            self.load_courses("data/courses.csv")

        # 1. Infer Level
        inferred_level = infer_user_level(user_query)
        
        # 2. Normalize/Expand Query
        expanded_query = normalize_query(user_query, self.abbr_map)

        debug_info = {
            "query": user_query,
            "expanded_query": expanded_query,
            "inferred_level": inferred_level,
            "pre_filter_count": 0,
            "total_courses": len(self.courses_df) if self.courses_df is not None else 0,
            "top_raw_scores": [],
            "keyword_warning": None
        }

        if not user_query.strip():
            return {"results": [], "debug_info": debug_info}

        # 3. Keyword Guardrail with filler word filtering
        clean_q = expanded_query.lower()
        query_words = set(re.findall(r'\b\w+\b', clean_q))
        
        filler_words = {
            'i', 'im', 'me', 'my', 'we', 'you', 'your', 'want', 'learn', 'need', 'looking', 
            'interested', 'course', 'courses', 'training', 'tutorial', 'search', 'find', 'get', 'take',
            'please', 'pls', 'can', 'could', 'would', 'should',
            'in', 'of', 'for', 'and', 'with', 'a', 'the', 'to', 'on', 'at', 'by', 'from', 'about',
            'beginner', 'intermediate', 'advanced', 'level',
            'عاوز', 'محتاج', 'اتعلم', 'كورس', 'شرح', 'ابحث', 'عن', 'في', 'من', 'على'
        }
        
        keywords = [w for w in query_words if w not in filler_words and len(w) > 2 and not w.isdigit()]
        
        if keywords:
            all_text_blob = " ".join(self.courses_df['combined_text'].tolist()).lower()
            missing_keywords = [kw for kw in keywords if kw not in all_text_blob]
            
            if missing_keywords:
                debug_info["keyword_warning"] = f"No courses found related to: {', '.join(missing_keywords)}"
                return {"results": [], "debug_info": debug_info}

        # 4. Apply Pre-Run Filters
        filtered_df = self.courses_df.copy()
        
        # Priority: Manual filter > Inferred level (if manual is "Any")
        target_level = pre_filters.get('level', "Any") if pre_filters else "Any"
        if target_level == "Any":
            target_level = inferred_level
            
        if target_level != "Any":
            # If level is "White", we treat it as Beginner for dataset filtering
            lookup_level = "Beginner" if target_level == "White" else target_level
            filtered_df = filtered_df[filtered_df['level'] == lookup_level]

        if pre_filters:
            if 'category' in pre_filters and pre_filters['category'] != "Any":
                filtered_df = filtered_df[filtered_df['category'] == pre_filters['category']]
            if 'max_duration' in pre_filters:
                 filtered_df = filtered_df[filtered_df['duration_hours'] <= pre_filters['max_duration']]

        debug_info["pre_filter_count"] = len(filtered_df)
        if filtered_df.empty:
            return {"results": [], "debug_info": debug_info}

        # 5. Semantic Search
        current_indices = filtered_df.index.tolist()
        results = []
        
        if self.model and self.embeddings is not None and len(self.embeddings) == len(self.courses_df):
            query_embedding = self.model.encode([expanded_query])
            subset_embeddings = self.embeddings[current_indices]
            similarities = cosine_similarity(query_embedding, subset_embeddings)[0]
            
            valid_mask = similarities >= similarity_threshold
            if not np.any(valid_mask):
                return {"results": [], "debug_info": debug_info}
            
            v_indices = np.where(valid_mask)[0]
            v_scores = similarities[valid_mask]
            
            sorted_idx = np.argsort(v_scores)[::-1][:top_k]
            top_local_idx = v_indices[sorted_idx]
            final_scores = v_scores[sorted_idx]
            
            debug_info["top_raw_scores"] = [float(s) for s in final_scores[:5]]
            
            # Safe Normalization into 1–10
            if len(final_scores) > 0:
                s_min, s_max = final_scores.min(), final_scores.max()
                for i, score in enumerate(final_scores):
                    course = filtered_df.iloc[top_local_idx[i]].to_dict()
                    course['similarity_score'] = float(score)
                    
                    if s_max == s_min:
                        rank = 10 if len(final_scores) == 1 else 5
                    else:
                        # Normalize to 1-10: (score - min) / (max - min) * 9 + 1
                        rank = int(((score - s_min) / (s_max - s_min)) * 9) + 1
                    
                    course['rank'] = rank
                    results.append(course)
        else:
            # Keyword fallback
            for _, row in filtered_df.iterrows():
                score = sum(1 for kw in keywords if kw in row['combined_text'].lower())
                if score > 0:
                    course = row.to_dict()
                    course['similarity_score'] = float(score)
                    course['rank'] = min(10, max(1, int(score)))
                    results.append(course)
            results = sorted(results, key=lambda x: x['similarity_score'], reverse=True)[:top_k]

        return {"results": results, "debug_info": debug_info}
