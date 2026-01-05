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
        self.model_name = model_name
        self.model = model
        self.courses_df = None
        self.embeddings = None
        self.abbr_map = {}
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
            map_path = os.path.join(cache_dir, f"abbr_map_{self.dataset_hash}.json")
            
            if os.path.exists(emb_path) and os.path.exists(map_path):
                self.embeddings = np.load(emb_path)
                with open(map_path, 'r') as f:
                    self.abbr_map = json.load(f)
                self.courses_df['combined_text'] = self.courses_df.apply(
                    lambda row: format_course_text(row, self.abbr_map), axis=1
                )
                self._initialize_model()
            else:
                self.abbr_map = build_abbreviation_map(self.courses_df)
                self._compute_embeddings()
                if self.embeddings is not None:
                    np.save(emb_path, self.embeddings)
                with open(map_path, 'w') as f:
                    json.dump(self.abbr_map, f)
                    
        except Exception as e:
            print(f"Error loading: {e}")

    def _compute_embeddings(self) -> None:
        if self.courses_df is None or self.courses_df.empty:
            return
        self.courses_df['combined_text'] = self.courses_df.apply(
                    lambda row: format_course_text(row, self.abbr_map), axis=1
                )
        self._initialize_model()
        if self.model:
            self.embeddings = self.model.encode(self.courses_df['combined_text'].tolist(), show_progress_bar=True)

    def recommend(
        self, 
        user_query: str, 
        top_k: int = 30, 
        pre_filters: Optional[Dict[str, Any]] = None,
        similarity_threshold: float = 0.2
    ) -> Dict[str, Any]:
        if self.courses_df is None:
            self.load_courses("data/courses.csv")

        # 1. Level Inference
        inferred_lvl = infer_user_level(user_query)
        
        # 2. Normalize Query (expansion)
        expanded_q = normalize_query(user_query, self.abbr_map)

        debug = {
            "query": user_query,
            "expanded_query": expanded_q,
            "inferred_level": inferred_lvl,
            "keyword_warning": None
        }

        if not user_query.strip():
            return {"results": [], "debug_info": debug}

        # 3. Guardrail with filler words
        q_words = set(re.findall(r'\b\w+\b', expanded_q.lower()))
        fillers = {
            'want', 'learn', 'need', 'looking', 'interested', 'course', 'training', 'i', 'the', 'to', 'in', 'on', 'at', 'by', 'from', 'with', 'and', 'or', 'a', 'an',
            'عاوز', 'محتاج', 'اتعلم', 'كورس', 'شرح', 'في', 'على', 'من', 'الى', 'عن'
        }
        keywords = [w for w in q_words if w not in fillers and len(w) > 2 and not w.isdigit()]
        
        if keywords:
            all_text = " ".join(self.courses_df['combined_text'].tolist()).lower()
            missing = [k for k in keywords if k not in all_text]
            if missing:
                debug["keyword_warning"] = f"No courses found related to: {', '.join(missing)}"
                return {"results": [], "debug_info": debug}

        # 4. Apply Pre-filters
        filtered_df = self.courses_df.copy()
        
        # Priority: select manually > inferred (if manual is Any)
        target_lvl = pre_filters.get('level', "Any") if pre_filters else "Any"
        if target_lvl == "Any":
            target_lvl = inferred_lvl
            
        if target_lvl != "Any":
            # Map "White" or specific inferred onto Beginner if needed, 
            # but usually we just look for exact match in 'level' column which we normalized
            lookup = "Beginner" if target_lvl == "White" else target_lvl
            filtered_df = filtered_df[filtered_df['level'] == lookup]

        if pre_filters:
            if 'category' in pre_filters and pre_filters['category'] != "Any":
                filtered_df = filtered_df[filtered_df['category'] == pre_filters['category']]
            if 'max_duration' in pre_filters:
                filtered_df = filtered_df[filtered_df['duration_hours'] <= pre_filters['max_duration']]

        if filtered_df.empty:
            return {"results": [], "debug_info": debug}

        # 5. Semantic Search
        indices = filtered_df.index.tolist()
        results = []
        
        if self.model and self.embeddings is not None:
            q_emb = self.model.encode([expanded_q])
            sims = cosine_similarity(q_emb, self.embeddings[indices])[0]
            
            mask = sims >= similarity_threshold
            if not np.any(mask):
                return {"results": [], "debug_info": debug}
            
            v_idx = np.where(mask)[0]
            v_scores = sims[mask]
            
            sort_idx = np.argsort(v_scores)[::-1][:top_k]
            top_local = v_idx[sort_idx]
            final_scores = v_scores[sort_idx]
            
            if len(final_scores) > 0:
                s_min, s_max = final_scores.min(), final_scores.max()
                for i, score in enumerate(final_scores):
                    course = filtered_df.iloc[top_local[i]].to_dict()
                    course['similarity_score'] = float(score)
                    # Strict 1-10
                    if s_max == s_min:
                        rank = 5
                    else:
                        rank = int(((score - s_min) / (s_max - s_min)) * 9) + 1
                    course['rank'] = rank
                    results.append(course)
        
        return {"results": results, "debug_info": debug}
