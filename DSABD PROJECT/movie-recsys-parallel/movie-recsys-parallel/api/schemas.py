from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


class Method(str, Enum):
    hybrid = "hybrid"
    als = "als"
    knn = "knn"


class MovieOut(BaseModel):
    movie_id: int
    title: str
    genres: List[str]
    year: Optional[int] = None
    n_ratings: int = 0
    in_model: bool = True


class SimilarMovieOut(MovieOut):
    similarity: float


class RecommendationOut(BaseModel):
    movie_id: int
    title: str
    genres: List[str]
    year: Optional[int] = None
    score: float = Field(description="Ranking score (higher is better); z-score blend for hybrid")
    predicted_rating: Optional[float] = Field(None, description="ALS-predicted star rating, 0.5-5")
    because: List[str] = Field(default_factory=list, description="Movies you rated highly that led to this pick")


class RatedMovie(BaseModel):
    """One movie the user has seen. Provide ``movie_id`` OR ``title``."""

    movie_id: Optional[int] = Field(None, examples=[1])
    title: Optional[str] = Field(None, examples=["Toy Story"])
    rating: float = Field(..., ge=0.5, le=5.0, examples=[5.0])

    @model_validator(mode="after")
    def _need_one(self):
        if self.movie_id is None and not self.title:
            raise ValueError("provide movie_id or title")
        return self


class RecommendRequest(BaseModel):
    ratings: List[RatedMovie] = Field(..., min_length=1, max_length=500)
    n: int = Field(10, ge=1, le=50)
    method: Method = Method.hybrid
    genres: Optional[List[str]] = Field(None, examples=[["Comedy", "Animation"]])
    min_ratings: int = Field(10, ge=0, description="ignore movies with fewer ratings than this in the training data")

    model_config = {
        "json_schema_extra": {
            "example": {
                "ratings": [
                    {"title": "Toy Story", "rating": 5},
                    {"title": "The Matrix", "rating": 4.5},
                    {"movie_id": 296, "rating": 2},
                ],
                "n": 10,
                "method": "hybrid",
            }
        }
    }


class RecommendResponse(BaseModel):
    method: Method
    recommendations: List[RecommendationOut]
    unresolved: List[str] = Field(default_factory=list, description="titles that could not be matched")
    matched: List[MovieOut] = Field(default_factory=list, description="movies matched from your input")


class UserRecommendResponse(BaseModel):
    user_id: int
    method: Method
    recommendations: List[RecommendationOut]


class HealthOut(BaseModel):
    status: str
    model: dict
