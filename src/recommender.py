import os
import json
import pandas as pd
import numpy as np
import re
from typing import List, Dict, Any, Optional, Union, Tuple
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
    get_dataset_hash,
    normalize_query,
    infer_user_level
)

class CourseRecommender:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', model: Any = None):
        self.model_name = model_name
        self.model = model
        self.courses_df = None
        self.embeddings = None
        self.dataset_hash = None
        
    def _initialize_model(self):
        if self.model is None and SentenceTransformer is not None:
            self.model = SentenceTransformer(self.model_name)

    def load_courses(self, source: Union[str, pd.DataFrame]) -> None:
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
            
            if os.path.exists(emb_path):
                self.embeddings = np.load(emb_path)
                self.courses_df['combined_text'] = self.courses_df.apply(format_course_text, axis=1)
                self._initialize_model()
            else:
                self._compute_embeddings()
                if self.embeddings is not None:
                    np.save(emb_path, self.embeddings)
                    
        except Exception as e:
            print(f"Error loading: {e}")

    def _compute_embeddings(self) -> None:
        if self.courses_df is None or self.courses_df.empty:
            return
        self.courses_df['combined_text'] = self.courses_df.apply(format_course_text, axis=1)
        self._initialize_model()
        if self.model:
            self.embeddings = self.model.encode(self.courses_df['combined_text'].tolist(), show_progress_bar=True)

    def recommend(
        self, 
        user_query: str, 
        top_k: int = 30, 
        pre_filters: Optional[Dict[str, Any]] = None,
        similarity_threshold: float = 0.1
    ) -> Dict[str, Any]:
        """
        Get recommendations with STRICT keyword matching rule.
        """
        if self.courses_df is None:
            self.load_courses("data/courses.csv")

        # 1. Normalize Query
        cleaned_query = normalize_query(user_query)
        # Strong keywords (length >= 3)
        strong_keywords = [t for t in cleaned_query.split() if len(t) >= 3]

        debug = {
            "cleaned_query": cleaned_query,
            "strong_keywords": strong_keywords,
            "error_message": None
        }

        if not strong_keywords:
            return {"results": [], "debug_info": debug}

        # 2. HARD KEYWORD FILTER
        # STRICT RULE: A course matches ONLY if EVERY strong keyword exists in (title|skills|description)
        match_mask = pd.Series([True] * len(self.courses_df))
        for kw in strong_keywords:
            kw_pat = re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE)
            # Course-wise check: keyword must be in at least one of the 3 fields
            kw_match = (
                self.courses_df['title'].str.contains(kw_pat, na=False) |
                self.courses_df['skills'].str.contains(kw_pat, na=False) |
                self.courses_df['description'].str.contains(kw_pat, na=False)
            )
            match_mask &= kw_match # This already implements AND logic for keywords

        filtered_indices = self.courses_df.index[match_mask].tolist()
        
        if not filtered_indices:
            debug["error_message"] = f"❌ No courses found related to: {', '.join(strong_keywords)}. Please try another keyword."
            return {"results": [], "debug_info": debug}

        # 3. Apply Pre-filters ONLY on the matched keyword subset
        subset_df = self.courses_df.loc[filtered_indices].copy()
        
        if pre_filters:
            if 'level' in pre_filters and pre_filters['level'] != "Any" and pre_filters['level'] != "White":
                subset_df = subset_df[subset_df['level'] == pre_filters['level']]
            if 'category' in pre_filters and pre_filters['category'] != "Any":
                subset_df = subset_df[subset_df['category'] == pre_filters['category']]
            if 'max_duration' in pre_filters:
                subset_df = subset_df[subset_df['duration_hours'] <= pre_filters['max_duration']]

        if subset_df.empty:
            return {"results": [], "debug_info": debug}

        # 4. Semantic Search ONLY within valid subset
        indices = subset_df.index.tolist()
        results = []
        
        if self.model and self.embeddings is not None:
            q_emb = self.model.encode([cleaned_query])
            sims = cosine_similarity(q_emb, self.embeddings[indices])[0]
            
            # Since we already passed the hard filter, we can be more lenient with similarity
            mask = sims >= similarity_threshold
            v_idx = np.where(mask)[0]
            v_scores = sims[mask]
            
            sort_idx = np.argsort(v_scores)[::-1][:top_k]
            top_local = v_idx[sort_idx]
            final_scores = v_scores[sort_idx]
            
            if len(final_scores) > 0:
                s_min, s_max = final_scores.min(), final_scores.max()
                for i, score in enumerate(final_scores):
                    course = subset_df.iloc[top_local[i]].to_dict()
                    course['similarity_score'] = float(score)
                    # Rank 1-10
                    if s_max == s_min:
                        rank = 5
                    else:
                        rank = int(((score - s_min) / (s_max - s_min)) * 9) + 1
                    course['rank'] = rank
                    results.append(course)

        return {"results": results, "debug_info": debug}
