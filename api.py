"""
FastAPI endpoint for Course Recommender System.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
import uvicorn
import os

from src.recommender import CourseRecommender

app = FastAPI(
    title="Zedny Course Recommender API",
    description="AI-Powered Course Recommendation System",
    version="1.3.0"
)

recommender = None

def get_recommender():
    global recommender
    if recommender is None:
        recommender = CourseRecommender()
        recommender.load_courses("data/courses.csv")
    return recommender

class RecommendRequest(BaseModel):
    query: str = Field(..., min_length=2)
    top_k: int = 30
    filters: Optional[Dict[str, Any]] = None

class RecommendResponse(BaseModel):
    timestamp: str
    inferred_level: str
    results: List[Dict[str, Any]]
    debug_info: Dict[str, Any]

@app.get("/health")
def health():
    rec = get_recommender()
    return {
        "status": "healthy",
        "dataset_size": len(rec.courses_df) if rec.courses_df is not None else 0,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest):
    try:
        rec = get_recommender()
        response = rec.recommend(
            user_query=request.query,
            top_k=request.top_k,
            pre_filters=request.filters
        )
        return RecommendResponse(
            timestamp=datetime.now().isoformat(),
            inferred_level=response["debug_info"]["inferred_level"],
            results=response["results"],
            debug_info=response["debug_info"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
