# Parallel Movie Recommender

Collaborative-filtering movie recommendations, accelerated with **OpenMP multithreading** and served through a **FastAPI** service.

Item–item similarity and ALS matrix factorisation are distributed across CPU cores instead of running sequentially. A benchmark harness compares **1 / 2 / 4 / 8 threads** (time, speed-up, efficiency) and verifies that parallel runs produce *exactly* the same model as the sequential one.

```
 ratings.csv ──► [ C++ / OpenMP trainer ] ──► artifacts/ ──► [ FastAPI service ] ──► user
 movies.csv          │                          sim_idx/val        /recommend
                     ├─ 1. load + CSR/CSC       user/item factors  /users/{id}/recommendations
                     ├─ 2. item-item similarity (parallel)         /movies/{id}/similar
                     └─ 3. ALS-WR               (parallel)         /movies/search
```

> ## ⚠️ About the data
> The provided `movies.csv` (10,329 movies) is only half of MovieLens: collaborative filtering also needs the **user–movie ratings** (`ratings.csv`)

---

## Quick start

Requirements: `g++` with OpenMP (GCC ≥ 9 / Clang + libomp), Python ≥ 3.10.

```bash
pip install -r requirements.txt

make train        # compile the C++ core, (generate demo ratings if absent), train, write artifacts/
make api          # FastAPI on http://localhost:8000   (UI at /, Swagger at /docs)
make benchmark    # 1/2/4/8-thread comparison -> results/
make test         # unit tests
```

Docker: `docker build -t recsys . && docker run -p 8000:8000 recsys`

Useful knobs: `make train THREADS=8 HOLDOUT=0.1` (prints held-out RMSE), or call the binary directly:

```bash
./cpp/recsys --ratings data/ratings.csv --out artifacts --threads 8 \
             --rank 32 --iters 10 --lambda 0.05 --topk 50 --holdout 0.1
```

## API

| Method & path | Purpose |
|---|---|
| `POST /recommend` | **Main user-input endpoint.** Send a few rated movies (by `title` or `movie_id`), get personalised recommendations. Works for brand-new users. |
| `GET /users/{user_id}/recommendations` | Recommendations for an existing user (`?n=&method=&genres=`) |
| `GET /movies/search?q=matrix` | Title search (case/article-insensitive, typo tolerant) |
| `GET /movies/{id}` · `GET /movies/{id}/similar` | Movie info · most similar movies |
| `GET /health` · `GET /benchmark` | Model info · last benchmark results |

```bash
curl -X POST localhost:8000/recommend -H 'Content-Type: application/json' -d '{
  "ratings": [
    {"title": "Toy Story", "rating": 5},
    {"title": "The Matrix", "rating": 4.5},
    {"movie_id": 296, "rating": 2}
  ],
  "n": 10, "method": "hybrid", "genres": ["Comedy"]
}'
```

Each recommendation contains `movie_id, title, genres, year, score, predicted_rating, because` (the movies you rated highly that led to the pick). Titles that cannot be matched are returned in `unresolved`. `method` is `hybrid` (default), `als`, or `knn`. A small browser UI is served at `/`.

## How the parallelism works

**Item–item similarity** (adjusted cosine, user-mean-centred, with significance shrinkage `co/(co+10)`):

* The similarity matrix is computed one *row* (item) at a time. Rows are independent → embarrassingly parallel.
* `#pragma omp for schedule(dynamic, 8)` over items. Dynamic scheduling matters because cost is heavily skewed: a blockbuster touches far more co-ratings than a niche film.
* Every thread owns a **private sparse accumulator** (dense `dot[]`/`co[]` arrays plus a "touched" list that is cleared after each row), so there are no locks and no false sharing on shared data.
* Each thread keeps the top-K neighbours and writes them into its own **disjoint slice** of the final matrix → the "merge" is lock-free by construction.

**ALS-WR matrix factorisation:**

* Fix item factors → every user solves an independent `k×k` regularised normal-equation system (Cholesky). Then fix user factors → same for items. Each half-step is a parallel loop over rows.
* Complexity per iteration ≈ `O(nnz·k² + (U+I)·k³)`; the `nnz·k²` Gram-matrix accumulation dominates and parallelises perfectly.

**What stays sequential (Amdahl):** CSV parsing / CSR construction and artifact writing. These are reported separately (`time_load_s`) and excluded from the compute speed-up, so the benchmark isolates the parallel stages while keeping the serial fraction visible. On large inputs, parallel parsing would be the next optimisation.

**Correctness guarantees (tested):** `tests/test_core.py` checks that (a) the C++ similarity equals an independent NumPy brute-force implementation, (b) 1-thread and 4-thread runs give **bit-identical** similarity matrices and ALS factors (every row is solved by exactly one thread, with deterministic initialisation), and (c) ALS beats the global-mean baseline on held-out ratings.

## Serving

* **Item-kNN** score for an item = shrunk, similarity-weighted average of the user's mean-centred ratings over the item's neighbours.
* **ALS** score = `μ + u·v`. For new users, `u` is obtained by a ridge "fold-in" using the same objective as training, so no retraining is needed.
* **Hybrid** = 0.6·z(ALS) + 0.4·z(kNN). Items with fewer than `min_ratings` (default 10) training ratings are excluded to avoid noisy long-tail picks; already-seen movies are never recommended.

## Benchmark results

`make benchmark` writes `results/benchmark.{md,csv,json}` and `results/speedup.png`. The harness reports the median of 3 runs per thread count, the machine's available core count, and flags any thread count exceeding it.

Result obtained while building this project (668 users × 10,147 items, 98.5k synthetic ratings, rank 32, 10 ALS iterations):

| Threads | Similarity (s) | ALS (s) | Total (s) | Speed-up |
|---:|---:|---:|---:|---:|
| 1 | 0.368 | 0.687 | 1.065 | 1.00× |
| 2 \* | 0.350 | 0.694 | 1.044 | 1.02× |
| 4 \* | 0.362 | 0.696 | 1.058 | 1.01× |
| 8 \* | 0.364 | 0.716 | 1.080 | 0.99× |

\* **This run happened in a single-core container, so no speed-up was possible.** These numbers only demonstrate that the harness works and that the result is identical for every thread count. **They are not evidence of scaling.** Run `make benchmark` on a multi-core machine for real numbers (and `make bench-large`, ~4M ratings, because the small MovieLens set finishes in about a second and is dominated by thread start-up overhead). Fill your measured table here for your report.

Held-out accuracy on the synthetic data: test RMSE 0.749 vs 0.970 for the global-mean baseline (this validates the ALS implementation, not MovieLens performance).

## Project layout

```
cpp/recsys.cpp                      OpenMP trainer (similarity + ALS)
api/{main,engine,schemas}.py        FastAPI app, recommendation engine, request/response models
api/static/index.html               tiny browser UI
benchmarks/run_benchmark.py         1/2/4/8-thread benchmark, speed-up/efficiency, plot
scripts/generate_synthetic_ratings.py   demo ratings (only if real ones are missing)
tests/                              core (C++ vs NumPy), engine and API tests
data/movies.csv                     your MovieLens movies file
```

## Limitations / next steps

* Similarity is held as top-K per item (default 50), not the full dense matrix, which keeps memory O(items·K) and is what makes large catalogues practical.
* The cold-start path needs at least one rated movie that exists in the training data.
* Possible extensions: implicit-feedback ALS, parallel CSV parsing, ranking metrics (Precision@K / NDCG) in the evaluation, periodic retraining.
