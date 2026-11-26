#!/usr/bin/env python3
"""Generate SYNTHETIC MovieLens-style ratings for the movies in data/movies.csv.

Use this ONLY when the real MovieLens ``ratings.csv`` is not available.
The data has realistic structure (long-tail popularity, user/item biases,
genre-driven taste) so collaborative filtering has real signal to learn, but it
is *not* real user behaviour: do not report its accuracy as MovieLens results.

Scale knobs (--users / --mean-ratings) let you build much larger problems to
benchmark parallel scalability.

Example:
    python scripts/generate_synthetic_ratings.py --users 668 --mean-ratings 160
    python scripts/generate_synthetic_ratings.py --users 20000 --mean-ratings 200 \
        --out data/ratings_large.csv
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--movies", default="data/movies.csv")
    ap.add_argument("--out", default="data/ratings.csv")
    ap.add_argument("--users", type=int, default=668, help="number of users (668 ~ MovieLens latest-small)")
    ap.add_argument("--mean-ratings", type=float, default=160, help="mean ratings per user")
    ap.add_argument("--latent-dim", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    movies = pd.read_csv(args.movies)
    n_movies = len(movies)
    genres = sorted({g for s in movies["genres"] for g in s.split("|") if g != "(no genres listed)"})
    gidx = {g: k for k, g in enumerate(genres)}
    G = np.zeros((n_movies, len(genres)), dtype=np.float32)
    for r, s in enumerate(movies["genres"]):
        for g in s.split("|"):
            if g in gidx:
                G[r, gidx[g]] = 1.0
    G /= np.maximum(G.sum(1, keepdims=True), 1.0)

    d = args.latent_dim
    W = rng.normal(0, 1.0, (len(genres), d)).astype(np.float32)           # genre -> taste space
    Q = G @ W + rng.normal(0, 0.35, (n_movies, d)).astype(np.float32)      # movie factors
    item_bias = rng.normal(0, 0.45, n_movies).astype(np.float32)

    # long-tail popularity (Zipf-like over a random permutation of movies)
    rank = rng.permutation(n_movies) + 1
    pop = 1.0 / np.power(rank, 0.85)
    log_pop = np.log(pop / pop.sum())

    P = rng.normal(0, 0.55, (args.users, d)).astype(np.float32)           # user taste
    user_bias = rng.normal(0, 0.4, args.users).astype(np.float32)
    counts = np.clip(rng.lognormal(np.log(args.mean_ratings) - 0.5, 1.0, args.users), 20, n_movies // 2).astype(int)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    total = 0
    with out.open("w") as f:
        f.write("userId,movieId,rating,timestamp\n")
        for u0 in range(0, args.users, 500):
            rows = []
            for u in range(u0, min(u0 + 500, args.users)):
                # Gumbel top-k == weighted sampling without replacement, much faster than rng.choice
                keys = log_pop + rng.gumbel(size=n_movies)
                # users drift toward their own taste: boost items they would like
                keys += 0.6 * (Q @ P[u])
                items = np.argpartition(-keys, counts[u])[: counts[u]]
                raw = 3.4 + user_bias[u] + item_bias[items] + 0.55 * (Q[items] @ P[u]) + rng.normal(0, 0.55, len(items))
                rating = np.clip(np.round(raw * 2) / 2, 0.5, 5.0)
                ts = rng.integers(8.0e8, 1.6e9, len(items))
                rows.append(
                    pd.DataFrame(
                        {"userId": u + 1, "movieId": movies["movieId"].to_numpy()[items], "rating": rating, "timestamp": ts}
                    )
                )
            chunk = pd.concat(rows)
            chunk.to_csv(f, header=False, index=False)
            total += len(chunk)
    print(f"[synthetic] wrote {total:,} ratings for {args.users:,} users x {n_movies:,} movies -> {out} "
          f"({time.time() - t0:.1f}s)")
    print("[synthetic] NOTE: synthetic data. Replace data/ratings.csv with the real MovieLens file for real results.")


if __name__ == "__main__":
    main()
