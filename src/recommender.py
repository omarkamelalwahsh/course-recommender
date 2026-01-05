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

    def _get_keyword_score(self, query_tokens: List[str], row: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Calculate keyword match score based on weights."""
        if not query_tokens:
            return 0.0, []
        
        matches = []
        score = 0.0
        
        title_text = str(row['title']).lower()
        skills_text = str(row['skills']).lower()
        desc_text = str(row['description']).lower()
        
        for t in query_tokens:
            found = False
            if t in title_text:
                score += 1.0
                found = True
            elif t in skills_text:
                score += 0.5
                found = True
            elif t in desc_text:
                score += 0.2
                found = True
            
            if found:
                matches.append(t)
        
        # Normalize: total possible score is query_tokens len * 1.0 (assuming top matches in title)
        norm_score = min(score / len(query_tokens), 1.0)
        return norm_score, matches

    def recommend(
        self, 
        user_query: str, 
        top_k: int = 30, 
        pre_filters: Optional[Dict[str, Any]] = None,
        similarity_threshold: float = 0.15
    ) -> Dict[str, Any]:
        if self.courses_df is None:
            self.load_courses("data/courses.csv")

        # 1. Level Inference
        inferred_lvl = infer_user_level(user_query)
        
        # 2. Clean/Normalize Query
        cleaned_query = normalize_query(user_query, self.abbr_map)
        strong_keywords = [t for t in cleaned_query.split() if len(t) >= 3]

        debug = {
            "original_query": user_query,
            "cleaned_query": cleaned_query,
            "strong_keywords": strong_keywords,
            "inferred_level": inferred_lvl,
            "keyword_warning": None
        }

        if not cleaned_query:
            return {"results": [], "debug_info": debug}

        # 3. Strong Keyword Guardrail
        if strong_keywords:
            all_text = " ".join(self.courses_df['combined_text'].tolist()).lower()
            missing = [k for k in strong_keywords if k not in all_text]
            if missing:
                debug["keyword_warning"] = f"No courses found related to: {missing[0]}"
                return {"results": [], "debug_info": debug}

        # 4. Filter
        filtered_df = self.courses_df.copy()
        target_lvl = pre_filters.get('level', "Any") if pre_filters else "Any"
        if target_lvl == "Any": target_lvl = inferred_lvl
        if target_lvl != "Any":
            lookup = "Beginner" if target_lvl == "White" else target_lvl
            filtered_df = filtered_df[filtered_df['level'] == lookup]

        if pre_filters:
            if 'category' in pre_filters and pre_filters['category'] != "Any":
                filtered_df = filtered_df[filtered_df['category'] == pre_filters['category']]
            if 'max_duration' in pre_filters:
                filtered_df = filtered_df[filtered_df['duration_hours'] <= pre_filters['max_duration']]

        if filtered_df.empty:
            return {"results": [], "debug_info": debug}

        # 5. Semantic Candidates
        indices = filtered_df.index.tolist()
        results = []
        
        if self.model and self.embeddings is not None:
            q_emb = self.model.encode([cleaned_query])
            sims = cosine_similarity(q_emb, self.embeddings[indices])[0]
            
            # Use lower threshold initially to catch candidates for hybrid scoring
            mask = sims >= similarity_threshold
            if not np.any(mask):
                return {"results": [], "debug_info": debug}
            
            v_idx = np.where(mask)[0]
            v_scores = sims[mask]
            
            # 6. Hybrid Scoring & Reranking
            candidates = []
            for i, local_idx in enumerate(v_idx):
                course_row = filtered_df.iloc[local_idx]
                k_score, matches = self._get_keyword_score(strong_keywords, course_row)
                
                sem_score = v_scores[i]
                final_score = 0.7 * sem_score + 0.3 * k_score
                
                # Penalty for zero keyword match in strong queries
                if k_score == 0 and strong_keywords:
                    final_score *= 0.3
                
                course_dict = course_row.to_dict()
                course_dict.update({
                    'semantic_score': float(sem_score),
                    'keyword_score': float(k_score),
                    'final_score': float(final_score),
                    'matched_tokens': matches
                })
                candidates.append(course_dict)
            
            # Sort by final score
            candidates = sorted(candidates, key=lambda x: x['final_score'], reverse=True)
            top_results = candidates[:top_k]
            
            # 7. Rank Normalization 1-10
            if top_results:
                f_scores = [c['final_score'] for c in top_results]
                s_min, s_max = min(f_scores), max(f_scores)
                for c in top_results:
                    if s_max == s_min:
                        c['rank'] = 5
                    else:
                        c['rank'] = int(((c['final_score'] - s_min) / (s_max - s_min)) * 9) + 1
                results = top_results

        return {"results": results, "debug_info": debug}
