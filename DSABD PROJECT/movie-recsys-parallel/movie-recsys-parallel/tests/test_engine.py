import unittest
from pathlib import Path

import pandas as pd

from api.engine import RecommenderEngine
from tests.helpers import BIN, ROOT, make_ratings, tmpdir, train


@unittest.skipUnless(BIN.exists(), "build first: make build")
class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tmpdir()
        d = Path(cls.tmp.name)
        movies = pd.read_csv(ROOT / "data" / "movies.csv").head(60)
        movies.to_csv(d / "movies.csv", index=False)
        make_ratings(d / "ratings.csv", movie_ids=movies["movieId"].tolist(), density=0.5)
        train(d / "ratings.csv", d / "art", threads=2, rank=8, iters=6, topk=10)
        cls.e = RecommenderEngine(str(d / "art"), str(d / "movies.csv"), str(d / "ratings.csv"))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_search(self):
        self.assertEqual(self.e.search("toy story", 1)[0]["movie_id"], 1)
        self.assertTrue(self.e.search("jumanjj", 3))  # typo tolerance

    def test_user_recs_exclude_seen_and_respect_n(self):
        seen = set(self.e._history[1][0].tolist())
        for method in ("hybrid", "als", "knn"):
            recs = self.e.recommend_for_user(1, n=5, method=method, min_ratings=0)
            self.assertLessEqual(len(recs), 5)
            self.assertTrue(recs)
            ids = {self.e.item_index[r.movie_id] for r in recs}
            self.assertFalse(ids & seen, method)

    def test_profile_recs_and_genre_filter(self):
        recs = self.e.recommend_for_profile({1: 5.0, 2: 4.0}, n=5, genres=["Comedy"], min_ratings=0)
        self.assertTrue(all("Comedy" in r.genres for r in recs))
        self.assertTrue(all(r.movie_id not in (1, 2) for r in recs))

    def test_similar_sorted_desc(self):
        sims = self.e.similar_movies(1, 5)
        vals = [s["similarity"] for s in sims]
        self.assertEqual(vals, sorted(vals, reverse=True))

    def test_unknown_inputs(self):
        with self.assertRaises(KeyError):
            self.e.recommend_for_user(999999)
        with self.assertRaises(ValueError):
            self.e.recommend_for_profile({-5: 4.0})


if __name__ == "__main__":
    unittest.main()
