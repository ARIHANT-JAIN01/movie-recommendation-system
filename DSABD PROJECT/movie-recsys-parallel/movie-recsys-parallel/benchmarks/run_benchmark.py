#!/usr/bin/env python3
"""Benchmark single- vs multi-threaded training (1, 2, 4, 8 threads by default).

For every thread count the trainer is run ``--repeats`` times; the MEDIAN of each
stage is reported. Writes into results/:
    benchmark.csv  benchmark.json  benchmark.md  speedup.png

Also verifies that the similarity matrix and ALS factors are bit-identical for
every thread count (parallelism must not change the answer).

    python benchmarks/run_benchmark.py --ratings data/ratings.csv
"""
import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "cpp" / "recsys"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def run_once(ratings: Path, threads: int, out: Path, extra: list) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    cmd = [str(BIN), "--ratings", str(ratings), "--out", str(out), "--threads", str(threads), *extra]
    subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)
    return json.loads((out / "meta.json").read_text())


def available_cores() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", default=str(ROOT / "data" / "ratings.csv"))
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--holdout", type=float, default=0.1)
    ap.add_argument("--results", default=str(ROOT / "results"))
    args = ap.parse_args()

    if not BIN.exists():
        sys.exit("cpp/recsys not built. Run `make build` first.")
    results = Path(args.results)
    results.mkdir(exist_ok=True)
    extra = ["--rank", str(args.rank), "--iters", str(args.iters), "--holdout", str(args.holdout)]
    cores = available_cores()

    rows, hashes = [], {}
    for t in args.threads:
        runs = []
        for rep in range(args.repeats):
            out = results / "runs" / f"t{t}"
            runs.append(run_once(Path(args.ratings), t, out, extra))
            print(f"  threads={t} rep={rep + 1}: sim={runs[-1]['time_similarity_s']:.3f}s als={runs[-1]['time_als_s']:.3f}s")
        hashes[t] = (sha(out / "sim_idx.i32"), sha(out / "sim_val.f32"), sha(out / "item_factors.f32"))
        med = lambda k: statistics.median(r[k] for r in runs)
        rows.append(
            {
                "threads": t,
                "similarity_s": med("time_similarity_s"),
                "als_s": med("time_als_s"),
                "compute_s": med("time_compute_s"),
                "load_s": med("time_load_s"),
                "test_rmse": runs[-1]["test_rmse"][-1],
            }
        )

    base = next((r for r in rows if r["threads"] == 1), rows[0])
    for r in rows:
        for k in ("similarity", "als", "compute"):
            r[f"speedup_{k}"] = base[f"{k}_s"] / r[f"{k}_s"]
        r["efficiency"] = r["speedup_compute"] / (r["threads"] / base["threads"])
    identical = len(set(hashes.values())) == 1

    meta = {
        "machine": platform.processor() or platform.machine(),
        "available_cores": cores,
        "python": platform.python_version(),
        "dataset": {k: runs[-1][k] for k in ("n_users", "n_items", "n_train")},
        "rank": args.rank,
        "iters": args.iters,
        "repeats": args.repeats,
        "outputs_identical_across_threads": identical,
        "oversubscribed_thread_counts": [r["threads"] for r in rows if r["threads"] > cores],
    }
    (results / "benchmark.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))

    import csv

    with (results / "benchmark.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    md = [
        f"### Benchmark ({meta['dataset']['n_users']:,} users x {meta['dataset']['n_items']:,} items, "
        f"{meta['dataset']['n_train']:,} ratings; rank={args.rank}, {args.iters} ALS iters; median of {args.repeats})",
        "",
        f"Cores available to this run: **{cores}**. Outputs identical across thread counts: **{identical}**.",
        "",
        "| Threads | Similarity (s) | ALS (s) | Total compute (s) | Speed-up | Efficiency |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        flag = " *" if r["threads"] > cores else ""
        md.append(
            f"| {r['threads']}{flag} | {r['similarity_s']:.3f} | {r['als_s']:.3f} | {r['compute_s']:.3f} | "
            f"{r['speedup_compute']:.2f}x | {r['efficiency'] * 100:.0f}% |"
        )
    if meta["oversubscribed_thread_counts"]:
        md += ["", f"\\* more threads than available cores ({cores}): speed-up cannot exceed ~1x here. "
                   "Re-run on a multi-core machine for meaningful scaling numbers."]
    (results / "benchmark.md").write_text("\n".join(md) + "\n")
    print("\n" + "\n".join(md))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        T = [r["threads"] for r in rows]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].bar([str(t) for t in T], [r["compute_s"] for r in rows], color="#4f46e5")
        ax[0].set_xlabel("threads")
        ax[0].set_ylabel("seconds (similarity + ALS)")
        ax[0].set_title("Execution time")
        ax[1].plot(T, [r["speedup_compute"] for r in rows], "o-", label="measured (total)", color="#4f46e5")
        ax[1].plot(T, [r["speedup_similarity"] for r in rows], "s--", label="similarity", color="#16a34a")
        ax[1].plot(T, [r["speedup_als"] for r in rows], "^--", label="ALS", color="#ea580c")
        ax[1].plot(T, T, ":", color="gray", label="ideal")
        ax[1].set_xlabel("threads")
        ax[1].set_ylabel("speed-up vs 1 thread")
        ax[1].set_title(f"Speed-up ({cores} core(s) available)")
        ax[1].legend()
        fig.tight_layout()
        fig.savefig(results / "speedup.png", dpi=140)
        print(f"plot -> {results / 'speedup.png'}")
    except ImportError:
        print("matplotlib not installed; skipping plot")


if __name__ == "__main__":
    main()
