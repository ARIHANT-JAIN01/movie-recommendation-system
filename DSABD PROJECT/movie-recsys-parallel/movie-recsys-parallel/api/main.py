"""FastAPI service for the parallel movie recommender.

Run:  uvicorn api.main:app --reload        (docs at http://127.0.0.1:8000/docs)
"""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .engine import RecommenderEngine
from .schemas import (
    HealthOut,
    Method,
    MovieOut,
    RecommendRequest,
    RecommendResponse,
    SimilarMovieOut,
    UserRecommendResponse,
)

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = os.getenv("RECSYS_ARTIFACTS", str(ROOT / "artifacts"))
MOVIES = os.getenv("RECSYS_MOVIES", str(ROOT / "data" / "movies.csv"))
RATINGS = os.getenv("RECSYS_RATINGS", str(ROOT / "data" / "ratings.csv"))
STATIC = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Model is loaded once at start-up; all requests then read it without locking.
    app.state.engine = RecommenderEngine(ARTIFACTS, MOVIES, RATINGS)
    yield


app = FastAPI(
    title="Parallel Movie Recommender API",
    version="1.0.0",
    description=(
        "Collaborative-filtering recommendations. Item-item similarity and ALS matrix factorisation "
        "are trained with an OpenMP-parallel C++ core; this service serves the results."
    ),
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


def engine(request: Request) -> RecommenderEngine:
    return request.app.state.engine


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC / "index.html")


@app.get("/health", response_model=HealthOut, tags=["meta"])
def health(request: Request):
    return {"status": "ok", "model": engine(request).info()}


@app.get("/benchmark", tags=["meta"], summary="Speed-up results from the last `make benchmark` run")
def benchmark():
    f = ROOT / "results" / "benchmark.json"
    if not f.exists():
        raise HTTPException(404, "No benchmark results yet. Run `make benchmark`.")
    return JSONResponse(json.loads(f.read_text()))


@app.get("/movies/search", response_model=List[MovieOut], tags=["movies"])
def search_movies(request: Request, q: str = Query(..., min_length=1, description="title text"), limit: int = Query(10, ge=1, le=50)):
    return engine(request).search(q, limit)


@app.get("/movies/{movie_id}", response_model=MovieOut, tags=["movies"])
def get_movie(request: Request, movie_id: int):
    m = engine(request).get_movie(movie_id)
    if m is None:
        raise HTTPException(404, f"movie {movie_id} not found")
    return m


@app.get("/movies/{movie_id}/similar", response_model=List[SimilarMovieOut], tags=["movies"])
def similar(request: Request, movie_id: int, n: int = Query(10, ge=1, le=50)):
    try:
        return engine(request).similar_movies(movie_id, n)
    except KeyError:
        raise HTTPException(404, f"movie {movie_id} has no similarity data (too few ratings or unknown id)")


@app.get("/users/{user_id}/recommendations", response_model=UserRecommendResponse, tags=["recommend"])
def recommend_for_user(
    request: Request,
    user_id: int,
    n: int = Query(10, ge=1, le=50),
    method: Method = Method.hybrid,
    genres: Optional[List[str]] = Query(None, description="repeat to filter: ?genres=Comedy&genres=Romance"),
    min_ratings: int = Query(10, ge=0),
):
    eng = engine(request)
    if not eng.known_user(user_id):
        raise HTTPException(404, f"user {user_id} not found; use POST /recommend for new users")
    recs = eng.recommend_for_user(user_id, n, method.value, genres, min_ratings)
    return {"user_id": user_id, "method": method, "recommendations": [r.as_dict() for r in recs]}


@app.post("/recommend", response_model=RecommendResponse, tags=["recommend"], summary="Recommendations for a new user from a few rated movies")
def recommend(request: Request, body: RecommendRequest):
    eng = engine(request)
    resolved, unresolved, matched = {}, [], []
    for item in body.ratings:
        mid = item.movie_id if item.movie_id is not None else eng.resolve_title(item.title or "")
        info = eng.get_movie(mid) if mid is not None else None
        if info is None or not info["in_model"]:
            unresolved.append(item.title or str(item.movie_id))
            continue
        resolved[mid] = item.rating
        matched.append(info)
    if not resolved:
        raise HTTPException(422, {"error": "none of the movies could be matched", "unresolved": unresolved})
    recs = eng.recommend_for_profile(resolved, body.n, body.method.value, body.genres, body.min_ratings)
    return {
        "method": body.method,
        "recommendations": [r.as_dict() for r in recs],
        "unresolved": unresolved,
        "matched": matched,
    }
