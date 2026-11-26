import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "cpp" / "recsys"


def make_ratings(path: Path, n_users=60, movie_ids=None, density=0.35, seed=3):
    """Low-rank-ish random ratings so there is real structure to learn."""
    rng = np.random.default_rng(seed)
    movie_ids = list(movie_ids if movie_ids is not None else range(1, 41))
    P, Q = rng.normal(size=(n_users, 3)), rng.normal(size=(len(movie_ids), 3))
    rows = []
    for u in range(n_users):
        for j, m in enumerate(movie_ids):
            if rng.random() < density:
                r = np.clip(np.round((3.3 + 0.6 * P[u] @ Q[j] + rng.normal(0, 0.3)) * 2) / 2, 0.5, 5)
                rows.append((u + 1, m, r))
    path.write_text("userId,movieId,rating,timestamp\n" + "\n".join(f"{u},{m},{r},0" for u, m, r in rows) + "\n")
    return rows


def train(ratings: Path, out: Path, threads=1, **kw):
    out.mkdir(parents=True, exist_ok=True)
    args = [str(BIN), "--ratings", str(ratings), "--out", str(out), "--threads", str(threads)]
    for k, v in kw.items():
        args += [f"--{k.rstrip('_').replace('_', '-')}", str(v)]
    subprocess.run(args, check=True, stderr=subprocess.DEVNULL)


def tmpdir():
    return tempfile.TemporaryDirectory()
