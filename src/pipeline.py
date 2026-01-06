import time
from typing import Dict, Any, List
from src.logger import setup_logger
from src.schemas import RecommendRequest, RecommendResponse, Recommendation
from src.data_loader import DataLoader
from src.ai.embeddings import EmbeddingService
from src.ai.gating import check_gating
from src.ai.ranker import normalize_rank_1_10
from src.utils import normalize_query, is_arabic
from src.config import (
    TOP_K_Candidates, 
    SEMANTIC_THRESHOLD_ARABIC, 
    SEMANTIC_THRESHOLD_GENERAL, 
    SEMANTIC_THRESHOLD_RELAXED
)

logger = setup_logger(__name__)

class CourseRecommenderPipeline:
    def __init__(self):
        self.data_loader = DataLoader()
        self.embedding_service = EmbeddingService()
        
        # Load data on init
        self.index, self.courses_df = self.data_loader.load_data()

    def recommend(self, request: RecommendRequest) -> RecommendResponse:
        start_time = time.time()
        
        if self.index is None or self.courses_df is None:
            return RecommendResponse(results=[], total_found=0, debug_info={"error": "Index missing"})

        # 1. Normalize Query
        original_query = request.query
        norm_query = normalize_query(original_query)
        is_ar = is_arabic(original_query)
        
        # Determine strictness per query type
        tokens = norm_query.split()
        is_short_query = len(tokens) <= 2
        
        # Base Threshold Selection
        if is_ar:
            current_threshold = SEMANTIC_THRESHOLD_ARABIC
        else:
            current_threshold = SEMANTIC_THRESHOLD_GENERAL

        logger.info(f"Query: '{original_query}' | Norm: '{norm_query}' | Short: {is_short_query} | Threshold: {current_threshold}")

        # 2. Embed Query
        query_vector = self.embedding_service.encode(norm_query)

        # 3. FAISS Search
        D, I = self.index.search(query_vector, TOP_K_Candidates)
        distances = D[0]
        indices = I[0]

        # 4. Filtering Strategy (Try Strict first, fallback if needed)
        
        def filter_candidates(threshold_val):
            candidates = []
            for i, idx in enumerate(indices):
                if idx == -1: continue 
                
                score = float(distances[i])
                course = self.courses_df.iloc[idx].to_dict()
                
                # Check Metadata Filters FIRST (Category/Level)
                if request.filters:
                    if request.filters.get('level') and request.filters['level'] != "Any":
                        if course.get('level') != request.filters['level']:
                            continue
                    if request.filters.get('category') and request.filters['category'] != "Any":
                        if course.get('category') != request.filters['category']:
                            continue

                # Apply Intelligent Gating
                is_valid, matched_kws = check_gating(
                    course=course,
                    score=score,
                    normalized_query=norm_query,
                    original_query=original_query,
                    threshold=threshold_val,
                    is_short_query=is_short_query
                )
                
                if is_valid:
                    # Enrich Data
                    title = course.get('title', '') or ""
                    desc = course.get('description', '') or ""
                    skills = course.get('skills', '') or ""
                    
                    why_reasons = []
                    if matched_kws:
                        why_reasons.append(f"Contains: {', '.join(matched_kws[:3])}")
                    if score > 0.6:
                        why_reasons.append("High Semantic Match")
                    elif score > 0.4:
                        why_reasons.append("Moderate Match")
                    
                    candidates.append({
                        "title": title,
                        "url": course.get('url', f"https://zedny.com/course/{course.get('course_id')}"), 
                        "score": score,
                        "description": desc,
                        "skills": skills,
                        "category": course.get('category', 'General'),
                        "level": course.get('level', 'All'),
                        "matched_keywords": matched_kws,
                        "why": why_reasons
                    })
            return candidates

        # Attempt 1: Standard/Strict
        valid_candidates = filter_candidates(current_threshold)
        
        # Attempt 2: Fallback (Relaxed) if results are too low (< 3)
        if len(valid_candidates) < 3 and not is_short_query:
            # We don't relax for Short Queries (Python must mean Python)
            # We only relax for long distinct queries like "how to lead a team effectively"
            logger.info("Low results, attempting relaxed threshold...")
            valid_candidates = filter_candidates(SEMANTIC_THRESHOLD_RELAXED)

        # 5. Reranking (Optional)
        if request.enable_reranking and len(valid_candidates) > 1:
            top_slice = valid_candidates[:20]
            tail_slice = valid_candidates[20:]
            
            titles = [c['title'] for c in top_slice]
            rerank_scores = self.embedding_service.rerank(norm_query, titles)
            
            for i, r_score in enumerate(rerank_scores):
                top_slice[i]['score'] = float(r_score)
                top_slice[i]['why'].insert(0, f"Reranker: {r_score:.2f}")
                
            top_slice.sort(key=lambda x: x['score'], reverse=True)
            valid_candidates = top_slice + tail_slice

        # 6. Rank Normalization & Formatting
        final_results = valid_candidates[:request.top_k]
        final_results = normalize_rank_1_10(final_results)
        
        output_list = []
        for res in final_results:
            rec = Recommendation(
                title=res['title'],
                url=res['url'],
                rank=res['rank'],
                score=res['score'], 
                category=res.get('category', 'General'),
                level=res.get('level', 'Any'),
                matched_keywords=res['matched_keywords'],
                why=res['why'],
                debug_info={
                    "desc_snippet": res['description'][:150]
                }
            )
            output_list.append(rec)

        elapsed = time.time() - start_time
        return RecommendResponse(
            results=output_list,
            total_found=len(output_list),
            debug_info={
                "time_taken": elapsed,
                "original_query": original_query,
                "normalized_query": norm_query,
                "is_short_query": is_short_query,
                "threshold_used": current_threshold
            }
        )
