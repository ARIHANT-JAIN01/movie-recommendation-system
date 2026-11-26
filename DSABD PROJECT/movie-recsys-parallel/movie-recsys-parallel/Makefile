# Parallel Movie Recommender - common tasks
PY       ?= python3
THREADS  ?= 0          # 0 = OpenMP default (all cores)
RATINGS  ?= data/ratings.csv
HOLDOUT  ?= 0          # e.g. HOLDOUT=0.1 to print test RMSE while training

.PHONY: all build data train benchmark bench-large api test clean

all: train

build:
	$(MAKE) -C cpp

data:
	@test -f $(RATINGS) || $(PY) scripts/generate_synthetic_ratings.py --out $(RATINGS)

train: build data
	./cpp/recsys --ratings $(RATINGS) --out artifacts --threads $(THREADS) --holdout $(HOLDOUT)

benchmark: build data
	$(PY) benchmarks/run_benchmark.py --ratings $(RATINGS)

# ~4M synthetic ratings: big enough for parallel scaling to show on a multi-core machine
bench-large: build
	$(PY) scripts/generate_synthetic_ratings.py --users 20000 --mean-ratings 200 --out data/ratings_large.csv
	$(PY) benchmarks/run_benchmark.py --ratings data/ratings_large.csv --results results/large

api: train
	uvicorn api.main:app --host 0.0.0.0 --port 8000

test: build
	$(PY) -m unittest discover -s tests -t . -v

clean:
	$(MAKE) -C cpp clean
	rm -rf results/runs artifacts/*.i32 artifacts/*.f32 artifacts/meta.json
