from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any

class Recommendation(BaseModel):
    title: str
    url: str
    category: Optional[str] = "General"
    level: Optional[str] = "All Levels"
    rank: int = Field(..., ge=1, le=10)
    score: float
    matched_keywords: List[str] = []
    why: List[str] = []
    debug_info: Optional[Dict[str, Any]] = None

class RecommendRequest(BaseModel):
    query: str = Field(..., min_length=2, description="User search query")
    top_k: int = Field(30, ge=1, le=100)
    filters: Optional[Dict[str, Any]] = None
    enable_reranking: bool = False

    @validator('query')
    def query_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty strings')
        return v

class RecommendResponse(BaseModel):
    results: List[Recommendation]
    total_found: int
    debug_info: Dict[str, Any]
