"""API tests. Skipped automatically when fastapi/httpx are not installed."""
import os
import unittest
from pathlib import Path

import pandas as pd

from tests.helpers import BIN, ROOT, make_ratings, tmpdir, train

try:
    from fastapi.testclient import TestClient

    HAVE = True
except Exception:  # pragma: no cover
    HAVE = False


@unittest.skipUnless(HAVE and BIN.exists(), "needs fastapi+httpx and a built trainer")
class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tmpdir()
        d = Path(cls.tmp.name)
        movies = pd.read_csv(ROOT / "data" / "movies.csv").head(60)
        movies.to_csv(d / "movies.csv", index=False)
        make_ratings(d / "ratings.csv", movie_ids=movies["movieId"].tolist(), density=0.5)
        train(d / "ratings.csv", d / "art", threads=2, rank=8, iters=5, topk=10)
        os.environ.update(RECSYS_ARTIFACTS=str(d / "art"), RECSYS_MOVIES=str(d / "movies.csv"), RECSYS_RATINGS=str(d / "ratings.csv"))
        from api.main import app

        cls.client = TestClient(app)
        cls.client.__enter__()  # runs lifespan

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)
        cls.tmp.cleanup()

    def test_health(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "ok")

    def test_search_and_similar(self):
        r = self.client.get("/movies/search", params={"q": "toy story"})
        self.assertEqual(r.json()[0]["movie_id"], 1)
        self.assertEqual(self.client.get("/movies/1/similar", params={"n": 3}).status_code, 200)
        self.assertEqual(self.client.get("/movies/9999999").status_code, 404)

    def test_user_recommendations(self):
        r = self.client.get("/users/1/recommendations", params={"n": 4, "method": "als", "min_ratings": 0})
        self.assertEqual(r.status_code, 200)
        self.assertLessEqual(len(r.json()["recommendations"]), 4)
        self.assertEqual(self.client.get("/users/424242/recommendations").status_code, 404)

    def test_post_recommend_by_title_and_id(self):
        body = {"ratings": [{"title": "Toy Story", "rating": 5}, {"movie_id": 2, "rating": 4}, {"title": "zzzzqqq", "rating": 3}],
                "n": 5, "min_ratings": 0}
        r = self.client.post("/recommend", json=body)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(len(r.json()["matched"]), 2)
        self.assertEqual(r.json()["unresolved"], ["zzzzqqq"])

    def test_validation(self):
        self.assertEqual(self.client.post("/recommend", json={"ratings": []}).status_code, 422)
        self.assertEqual(self.client.post("/recommend", json={"ratings": [{"title": "x", "rating": 9}]}).status_code, 422)
        self.assertEqual(self.client.post("/recommend", json={"ratings": [{"rating": 3}]}).status_code, 422)


if __name__ == "__main__":
    unittest.main()
