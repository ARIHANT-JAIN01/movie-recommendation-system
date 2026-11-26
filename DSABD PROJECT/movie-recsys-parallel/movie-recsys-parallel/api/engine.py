"""Serving-side recommendation engine.

Loads the artifacts written by the OpenMP trainer (``cpp/recsys``) and exposes:

* ``recommend_for_user``   - known MovieLens user id
* ``recommend_for_profile``- brand-new user described by a few rated movies (cold start)
* ``similar_movies``       - item-item neighbours
* ``search``               - title search

Scoring methods
---------------
``knn``     item-based CF on the parallel similarity matrix (adjusted cosine, top-K)
``als``     matrix-factorisation score  mu + u . v   (new users: ridge "fold-in")
``hybrid``  z-score blend of both (default)
"""
from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

_YEAR_RE = re.compile(r"\((\d{4})\)\s*$")


def _norm_title(t: str) -> str:
    t = _YEAR_RE.sub("", t).lower()
    # MovieLens stores "Matrix, The (1999)" -> "the matrix"
    m = re.match(r"^(.*), (the|a|an)(\s*\(.*\))?$", t.strip())
    if m:
        t = f"{m.group(2)} {m.group(1)}"
    return re.sub(r"[^a-z0-9 ]+", " ", t).strip()


@dataclass
class Recommendation:
    movie_id: int
    title: str
    genres: List[str]
    year: Optional[int]
    score: float
    predicted_rating: Optional[float]
    because: List[str]

    def as_dict(self) -> dict:
        return self.__dict__.copy()


class RecommenderEngine:
    def __init__(self, artifacts_dir: str, movies_csv: str, ratings_csv: Optional[str] = None):
        a = Path(artifacts_dir)
        meta_path = a / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"{meta_path} not found. Train first: `make train`")
        self.meta: dict = json.loads(meta_path.read_text())
        nU, nI = self.meta["n_users"], self.meta["n_items"]
        self.k = self.meta["rank"]
        self.K = self.meta["topk"]
        self.mu = float(self.meta["global_mean"])
        self.lam = float(self.meta["lambda"])
        self.rmin, self.rmax = float(self.meta["rating_min"]), float(self.meta["rating_max"])

        self.user_ids = np.fromfile(a / "user_ids.i32", dtype=np.int32)
        self.item_ids = np.fromfile(a / "item_ids.i32", dtype=np.int32)
        self.sim_idx = np.fromfile(a / "sim_idx.i32", dtype=np.int32).reshape(nI, self.K)
        self.sim_val = np.fromfile(a / "sim_val.f32", dtype=np.float32).reshape(nI, self.K)
        self.U = np.fromfile(a / "user_factors.f32", dtype=np.float32).reshape(nU, self.k)
        self.V = np.fromfile(a / "item_factors.f32", dtype=np.float32).reshape(nI, self.k)
        self.user_index: Dict[int, int] = {int(u): i for i, u in enumerate(self.user_ids)}
        self.item_index: Dict[int, int] = {int(m): i for i, m in enumerate(self.item_ids)}

        # ---- movie metadata -------------------------------------------------
        movies = pd.read_csv(movies_csv)
        movies["year"] = movies["title"].str.extract(_YEAR_RE)[0].astype("Int64")
        movies["genre_list"] = movies["genres"].apply(
            lambda s: [] if s == "(no genres listed)" else s.split("|")
        )
        movies["norm"] = movies["title"].map(_norm_title)
        self.movies = movies.set_index("movieId", drop=False)
        self._norm_titles = movies["norm"].tolist()
        self._norm_ids = movies["movieId"].tolist()
        self.item_genres: List[set] = [
            set(self.movies.at[int(m), "genre_list"]) if int(m) in self.movies.index else set() for m in self.item_ids
        ]

        # ---- user histories & item popularity --------------------------------
        self.item_count = np.zeros(nI, dtype=np.int32)
        self.item_mean = np.full(nI, self.mu, dtype=np.float32)
        self._history: Dict[int, tuple] = {}
        if ratings_csv and Path(ratings_csv).exists():
            r = pd.read_csv(ratings_csv, usecols=["userId", "movieId", "rating"])
            r = r[r["movieId"].isin(self.item_index.keys()) & r["userId"].isin(self.user_index.keys())]
            ii = r["movieId"].map(self.item_index).to_numpy()
            self.item_count = np.bincount(ii, minlength=nI).astype(np.int32)
            sums = np.bincount(ii, weights=r["rating"].to_numpy(), minlength=nI)
            nz = self.item_count > 0
            self.item_mean[nz] = (sums[nz] / self.item_count[nz]).astype(np.float32)
            for uid, g in r.groupby("userId"):
                self._history[int(uid)] = (
                    g["movieId"].map(self.item_index).to_numpy(),
                    g["rating"].to_numpy(dtype=np.float32),
                )

    # ------------------------------------------------------------------ info
    def info(self) -> dict:
        return {
            "n_users": int(self.meta["n_users"]),
            "n_items": int(self.meta["n_items"]),
            "n_movies_catalogue": int(len(self.movies)),
            "latent_factors": self.k,
            "neighbours_per_item": self.K,
            "trained_with_threads": self.meta["threads"],
            "time_similarity_s": self.meta["time_similarity_s"],
            "time_als_s": self.meta["time_als_s"],
            "test_rmse": (self.meta["test_rmse"] or [None])[-1],
            "baseline_test_rmse": self.meta.get("baseline_test_rmse"),
        }

    # ---------------------------------------------------------------- lookup
    def get_movie(self, movie_id: int) -> Optional[dict]:
        if movie_id not in self.movies.index:
            return None
        row = self.movies.loc[movie_id]
        return {
            "movie_id": int(movie_id),
            "title": row["title"],
            "genres": row["genre_list"],
            "year": None if pd.isna(row["year"]) else int(row["year"]),
            "n_ratings": int(self.item_count[self.item_index[movie_id]]) if movie_id in self.item_index else 0,
            "in_model": movie_id in self.item_index,
        }

    def search(self, query: str, limit: int = 10) -> List[dict]:
        q = _norm_title(query)
        if not q:
            return []
        scored = []
        for mid, t in zip(self._norm_ids, self._norm_titles):
            if q == t:
                s = 3.0
            elif t.startswith(q):
                s = 2.0 + len(q) / max(len(t), 1)
            elif q in t:
                s = 1.0 + len(q) / max(len(t), 1)
            else:
                continue
            m = self.item_index.get(mid)
            pop = np.log1p(self.item_count[m]) / 10 if m is not None else 0.0
            scored.append((s + pop, mid))
        if not scored:  # typo tolerance
            close = difflib.get_close_matches(q, self._norm_titles, n=limit, cutoff=0.6)
            ids = {t: i for t, i in zip(self._norm_titles, self._norm_ids)}
            scored = [(0.5, ids[c]) for c in close]
        scored.sort(reverse=True)
        return [self.get_movie(int(mid)) for _, mid in scored[:limit]]

    def resolve_title(self, title: str) -> Optional[int]:
        hits = self.search(title, limit=1)
        return hits[0]["movie_id"] if hits else None

    def known_user(self, user_id: int) -> bool:
        return user_id in self.user_index

    # --------------------------------------------------------------- scoring
    def _fold_in(self, items: np.ndarray, ratings: np.ndarray) -> np.ndarray:
        """Ridge solve for a user vector given a few ratings (same objective as ALS-WR)."""
        Vi = self.V[items].astype(np.float64)
        y = ratings.astype(np.float64) - self.mu
        A = Vi.T @ Vi + self.lam * len(items) * np.eye(self.k)
        return np.linalg.solve(A, Vi.T @ y)

    def _knn_scores(self, items: np.ndarray, ratings: np.ndarray):
        mean = float(ratings.mean())
        d = ratings - mean
        nb = self.sim_idx[items]                       # (n, K)
        w = self.sim_val[items]                        # (n, K)
        mask = nb >= 0
        flat_i = nb[mask]
        num = np.bincount(flat_i, weights=(w * d[:, None])[mask], minlength=len(self.item_ids))
        den = np.bincount(flat_i, weights=w[mask], minlength=len(self.item_ids))
        score = num / (den + 0.5)                      # shrink low-evidence items
        return score, den, mean

    @staticmethod
    def _z(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
        out = np.zeros_like(x, dtype=np.float64)
        v = x[valid]
        if len(v) > 1 and v.std() > 1e-9:
            out[valid] = (v - v.mean()) / v.std()
        return out

    def _recommend(
        self,
        items: np.ndarray,
        ratings: np.ndarray,
        user_vec: Optional[np.ndarray],
        n: int,
        method: str,
        genres: Optional[Sequence[str]],
        min_ratings: int,
        exclude: Iterable[int],
    ) -> List[Recommendation]:
        nI = len(self.item_ids)
        if self.item_count.sum() == 0:  # no ratings file supplied -> cannot filter by popularity
            min_ratings = 0
        valid = self.item_count >= min_ratings
        if len(items):
            valid[np.asarray(list(set(items.tolist()) | set(exclude)), dtype=int)] = False
        if genres:
            want = {g.lower() for g in genres}
            gmask = np.fromiter((bool(want & {x.lower() for x in g}) for g in self.item_genres), bool, nI)
            valid &= gmask

        als = (self.V @ user_vec + self.mu) if user_vec is not None else np.full(nI, self.mu)
        if len(items):
            knn, den, umean = self._knn_scores(items, ratings)
        else:
            knn, den, umean = np.zeros(nI), np.zeros(nI), self.mu
        knn_valid = valid & (den > 0)

        if method == "als":
            final = self._z(als, valid)
        elif method == "knn":
            final = self._z(knn, knn_valid)
            final[valid & ~knn_valid] = -10.0
        else:
            final = 0.6 * self._z(als, valid) + 0.4 * self._z(knn, knn_valid)
        final = np.where(valid, final, -np.inf)

        top = np.argpartition(-final, min(n, nI - 1))[:n]
        top = top[np.argsort(-final[top])]
        top = [t for t in top if np.isfinite(final[t])]

        reasons = self._reasons(items, ratings, top)
        out = []
        for t in top:
            mid = int(self.item_ids[t])
            m = self.get_movie(mid)
            out.append(
                Recommendation(
                    movie_id=mid,
                    title=m["title"],
                    genres=m["genres"],
                    year=m["year"],
                    score=round(float(final[t]), 4),
                    predicted_rating=round(float(np.clip(als[t], self.rmin, self.rmax)), 2) if user_vec is not None else None,
                    because=reasons.get(int(t), []),
                )
            )
        return out

    def _reasons(self, items: np.ndarray, ratings: np.ndarray, top: List[int], per_item: int = 2) -> Dict[int, List[str]]:
        """'Because you liked X' - the liked movies that contribute most similarity to each recommendation."""
        if not len(items) or not top:
            return {}
        d = ratings - ratings.mean()
        liked = [(j, dj) for j, dj in zip(items, d) if dj >= 0]
        best: Dict[int, List[tuple]] = {t: [] for t in top}
        top_set = set(top)
        for j, dj in liked:
            for nb, w in zip(self.sim_idx[j], self.sim_val[j]):
                if nb in top_set:
                    best[int(nb)].append((float(w) * (dj + 0.5), int(j)))
        out = {}
        for t, lst in best.items():
            lst.sort(reverse=True)
            out[t] = [self.movies.at[int(self.item_ids[j]), "title"] for _, j in lst[:per_item]]
        return out

    # ----------------------------------------------------------- public API
    def recommend_for_user(
        self, user_id: int, n: int = 10, method: str = "hybrid", genres: Optional[Sequence[str]] = None, min_ratings: int = 5
    ) -> List[Recommendation]:
        if user_id not in self.user_index:
            raise KeyError(user_id)
        ui = self.user_index[user_id]
        items, ratings = self._history.get(user_id, (np.array([], dtype=int), np.array([], dtype=np.float32)))
        return self._recommend(items, ratings, self.U[ui].astype(np.float64), n, method, genres, min_ratings, [])

    def recommend_for_profile(
        self,
        ratings_by_movie: Dict[int, float],
        n: int = 10,
        method: str = "hybrid",
        genres: Optional[Sequence[str]] = None,
        min_ratings: int = 5,
    ) -> List[Recommendation]:
        pairs = [(self.item_index[m], r) for m, r in ratings_by_movie.items() if m in self.item_index]
        if not pairs:
            raise ValueError("none of the supplied movies are in the trained model")
        items = np.array([p[0] for p in pairs], dtype=int)
        ratings = np.array([p[1] for p in pairs], dtype=np.float32)
        return self._recommend(items, ratings, self._fold_in(items, ratings), n, method, genres, min_ratings, [])

    def similar_movies(self, movie_id: int, n: int = 10) -> List[dict]:
        if movie_id not in self.item_index:
            raise KeyError(movie_id)
        i = self.item_index[movie_id]
        out = []
        for j, s in zip(self.sim_idx[i], self.sim_val[i]):
            if j < 0:
                break
            m = self.get_movie(int(self.item_ids[j]))
            out.append({**m, "similarity": round(float(s), 4)})
            if len(out) == n:
                break
        return out
