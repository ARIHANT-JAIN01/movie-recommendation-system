"""Validates the OpenMP trainer against an independent NumPy implementation."""
import hashlib
import json
import unittest
from pathlib import Path

import numpy as np

from tests.helpers import BIN, make_ratings, tmpdir, train


@unittest.skipUnless(BIN.exists(), "build first: make build")
class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tmpdir()
        cls.dir = Path(cls.tmp.name)
        cls.csv = cls.dir / "ratings.csv"
        cls.rows = make_ratings(cls.csv)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_similarity_matches_numpy_bruteforce(self):
        K, MIN_CO, SHRINK = 5, 3, 10.0
        out = self.dir / "t1"
        train(self.csv, out, threads=1, topk=K, min_co=MIN_CO, shrink=SHRINK, iters=1, rank=4)
        item_ids = np.fromfile(out / "item_ids.i32", dtype=np.int32)
        user_ids = sorted({r[0] for r in self.rows})
        ui = {u: k for k, u in enumerate(user_ids)}
        ii = {m: k for k, m in enumerate(item_ids)}
        R = np.zeros((len(user_ids), len(item_ids)))
        M = np.zeros_like(R, dtype=bool)
        for u, m, r in self.rows:
            R[ui[u], ii[m]], M[ui[u], ii[m]] = r, True
        mean = (R * M).sum(1) / M.sum(1)
        C = np.where(M, R - mean[:, None], 0.0)
        norm = np.sqrt((C**2).sum(0))
        co = M.T.astype(float) @ M.astype(float)
        S = (C.T @ C) / np.outer(norm, norm) * co / (co + SHRINK)
        S[co < MIN_CO] = 0
        np.fill_diagonal(S, 0)
        S[S < 0] = 0

        idx = np.fromfile(out / "sim_idx.i32", dtype=np.int32).reshape(-1, K)
        val = np.fromfile(out / "sim_val.f32", dtype=np.float32).reshape(-1, K)
        for i in range(len(item_ids)):
            expected = np.sort(S[i])[::-1][:K]
            expected = expected[expected > 0]
            got = val[i][idx[i] >= 0]
            np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-5, err_msg=f"item {i}")
            for j, s in zip(idx[i][idx[i] >= 0], got):  # indices must point at the right similarity
                self.assertAlmostEqual(float(S[i, j]), float(s), places=4)

    def test_results_identical_for_1_and_4_threads(self):
        digests = []
        for t in (1, 4):
            out = self.dir / f"det{t}"
            train(self.csv, out, threads=t, iters=3, rank=8, holdout=0.1)
            digests.append(
                {f: hashlib.sha256((out / f).read_bytes()).hexdigest() for f in
                 ("sim_idx.i32", "sim_val.f32", "user_factors.f32", "item_factors.f32")}
            )
        self.assertEqual(digests[0], digests[1], "parallel run must not change results")

    def test_als_beats_global_mean_baseline(self):
        out = self.dir / "als"
        train(self.csv, out, threads=2, iters=10, rank=8, holdout=0.15, lambda_=0.1)
        meta = json.loads((out / "meta.json").read_text())
        self.assertLess(meta["test_rmse"][-1], meta["baseline_test_rmse"])
        self.assertLess(meta["train_rmse"][-1], meta["train_rmse"][0])


if __name__ == "__main__":
    unittest.main()
