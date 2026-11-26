// =============================================================================
//  recsys.cpp  --  Parallel collaborative filtering engine (OpenMP)
//
//  Stages
//    1. Load ratings (userId,movieId,rating[,timestamp]) and build CSR/CSC.     [sequential]
//    2. Item-item adjusted-cosine similarity, top-K neighbours per item.        [parallel]
//         - each thread owns a private sparse accumulator and processes
//           dynamically-scheduled blocks of items (rows of the similarity matrix);
//         - results are written to disjoint slices of the final matrix, so
//           "merging" is lock-free.
//    3. ALS-WR matrix factorisation (Zhou et al., 2008).                        [parallel]
//         - user step and item step are both embarrassingly parallel:
//           every row solves an independent k x k normal-equation system.
//    4. Write binary artifacts + meta.json for the Python/FastAPI layer.
//
//  Build:  g++ -O3 -march=native -fopenmp -std=c++17 recsys.cpp -o recsys
// =============================================================================
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using std::vector;

// ----------------------------------------------------------------------------
// Utilities
// ----------------------------------------------------------------------------
static inline uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
static inline double u01(uint64_t x) { return (splitmix64(x) >> 11) * (1.0 / 9007199254740992.0); }

struct Config {
    std::string ratings, out = "artifacts";
    int threads = 0;       // 0 -> OpenMP default
    int rank = 32;         // ALS latent factors
    int iters = 10;        // ALS iterations
    double lambda = 0.05;  // ALS-WR regularisation
    int topk = 50;         // neighbours kept per item
    int min_co = 3;        // min co-raters for a valid item pair
    double shrink = 10.0;  // significance shrinkage
    double holdout = 0.0;  // fraction of ratings held out for RMSE
    uint64_t seed = 42;
};

struct CSR {
    int n = 0;
    vector<int64_t> ptr;
    vector<int> idx;
    vector<float> val;
};

static CSR build_csr(int nrows, const vector<int>& row, const vector<int>& col, const vector<float>& v) {
    CSR m;
    m.n = nrows;
    m.ptr.assign(nrows + 1, 0);
    for (int r : row) m.ptr[r + 1]++;
    for (int r = 0; r < nrows; ++r) m.ptr[r + 1] += m.ptr[r];
    m.idx.resize(row.size());
    m.val.resize(row.size());
    vector<int64_t> pos(m.ptr.begin(), m.ptr.end() - 1);
    for (size_t k = 0; k < row.size(); ++k) {
        int64_t p = pos[row[k]]++;
        m.idx[p] = col[k];
        m.val[p] = v[k];
    }
    return m;
}

static void write_bin(const std::string& path, const void* data, size_t bytes) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {
        std::fprintf(stderr, "cannot write %s (does the output directory exist?)\n", path.c_str());
        std::exit(2);
    }
    if (bytes) std::fwrite(data, 1, bytes, f);
    std::fclose(f);
}

// ----------------------------------------------------------------------------
// Stage 1: loading
// ----------------------------------------------------------------------------
struct Raw {
    vector<int> uid, iid;
    vector<float> r;
};

static bool load_ratings(const std::string& path, Raw& raw) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::string buf(sz, '\0');
    if (sz > 0 && std::fread(&buf[0], 1, sz, f) != (size_t)sz) {
        std::fclose(f);
        return false;
    }
    std::fclose(f);

    const char* p = buf.c_str();
    const char* end = p + sz;
    // skip header line if the first char is not a digit
    if (p < end && !(*p >= '0' && *p <= '9')) {
        while (p < end && *p != '\n') ++p;
    }
    while (p < end) {
        while (p < end && (*p == '\n' || *p == '\r' || *p == ' ')) ++p;
        if (p >= end) break;
        char* q;
        long u = std::strtol(p, &q, 10);
        if (*q != ',') {
            while (p < end && *p != '\n') ++p;
            continue;
        }
        p = q + 1;
        long i = std::strtol(p, &q, 10);
        if (*q != ',') {
            while (p < end && *p != '\n') ++p;
            continue;
        }
        p = q + 1;
        double r = std::strtod(p, &q);
        p = q;
        while (p < end && *p != '\n') ++p;  // ignore timestamp / extra columns
        raw.uid.push_back((int)u);
        raw.iid.push_back((int)i);
        raw.r.push_back((float)r);
    }
    return !raw.r.empty();
}

// Map sparse external ids -> dense, ascending index (deterministic).
static void densify(const vector<int>& ids, vector<int>& dense, vector<int>& table_ids) {
    int mx = 0;
    for (int v : ids) mx = std::max(mx, v);
    vector<int> map((size_t)mx + 1, -1);
    for (int v : ids) map[v] = 0;
    for (int v = 0; v <= mx; ++v)
        if (map[v] == 0) {
            map[v] = (int)table_ids.size();
            table_ids.push_back(v);
        }
    dense.resize(ids.size());
    for (size_t k = 0; k < ids.size(); ++k) dense[k] = map[ids[k]];
}

// ----------------------------------------------------------------------------
// Stage 2: parallel item-item adjusted cosine similarity
// ----------------------------------------------------------------------------
static void item_similarity(const CSR& byUser /*centered*/, const CSR& byItem /*centered*/, const Config& cfg,
                            vector<int>& out_idx, vector<float>& out_val, double& pairs_visited) {
    const int nI = byItem.n, K = cfg.topk;
    out_idx.assign((size_t)nI * K, -1);
    out_val.assign((size_t)nI * K, 0.f);

    vector<double> norm(nI, 0.0);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < nI; ++i) {
        double s = 0;
        for (int64_t p = byItem.ptr[i]; p < byItem.ptr[i + 1]; ++p) s += (double)byItem.val[p] * byItem.val[p];
        norm[i] = std::sqrt(s);
    }

    double visited = 0;
#pragma omp parallel reduction(+ : visited)
    {
        // Thread-private sparse accumulator: no sharing, no locks.
        vector<double> dot(nI, 0.0);
        vector<int> co(nI, 0);
        vector<int> touched;
        vector<std::pair<float, int>> cand;

        // Items have very skewed cost (popular movies touch many users) -> dynamic schedule.
#pragma omp for schedule(dynamic, 8)
        for (int i = 0; i < nI; ++i) {
            touched.clear();
            for (int64_t p = byItem.ptr[i]; p < byItem.ptr[i + 1]; ++p) {
                const int u = byItem.idx[p];
                const double ri = byItem.val[p];
                for (int64_t q = byUser.ptr[u]; q < byUser.ptr[u + 1]; ++q) {
                    const int j = byUser.idx[q];
                    if (j == i) continue;
                    if (co[j] == 0) touched.push_back(j);
                    ++co[j];
                    dot[j] += ri * byUser.val[q];
                }
                visited += (double)(byUser.ptr[u + 1] - byUser.ptr[u]);
            }
            cand.clear();
            const double ni = norm[i];
            for (int j : touched) {
                if (co[j] >= cfg.min_co && ni > 0 && norm[j] > 0) {
                    double s = dot[j] / (ni * norm[j]);
                    s *= co[j] / (co[j] + cfg.shrink);
                    if (s > 0) cand.emplace_back((float)s, j);
                }
                dot[j] = 0;
                co[j] = 0;
            }
            const int keep = std::min<int>(K, (int)cand.size());
            std::partial_sort(cand.begin(), cand.begin() + keep, cand.end(), [](const auto& a, const auto& b) {
                return a.first > b.first || (a.first == b.first && a.second < b.second);
            });
            for (int t = 0; t < keep; ++t) {  // disjoint slice => lock-free "merge"
                out_idx[(size_t)i * K + t] = cand[t].second;
                out_val[(size_t)i * K + t] = cand[t].first;
            }
        }
    }
    pairs_visited = visited;
}

// ----------------------------------------------------------------------------
// Stage 3: parallel ALS-WR
// ----------------------------------------------------------------------------
static bool cholesky_solve(double* A, double* b, int k) {
    for (int j = 0; j < k; ++j) {
        double s = A[j * k + j];
        for (int p = 0; p < j; ++p) s -= A[j * k + p] * A[j * k + p];
        if (s <= 1e-12) return false;
        const double d = std::sqrt(s);
        A[j * k + j] = d;
        for (int i = j + 1; i < k; ++i) {
            double t = A[i * k + j];
            for (int p = 0; p < j; ++p) t -= A[i * k + p] * A[j * k + p];
            A[i * k + j] = t / d;
        }
    }
    for (int i = 0; i < k; ++i) {  // L y = b
        double t = b[i];
        for (int p = 0; p < i; ++p) t -= A[i * k + p] * b[p];
        b[i] = t / A[i * k + i];
    }
    for (int i = k - 1; i >= 0; --i) {  // L^T x = y
        double t = b[i];
        for (int p = i + 1; p < k; ++p) t -= A[p * k + i] * b[p];
        b[i] = t / A[i * k + i];
    }
    return true;
}

// Solve every row of X given the fixed factors F.  Rows are independent => parallel.
static void als_half_step(const CSR& m, const vector<float>& F, vector<float>& X, int k, double lambda, double mu) {
#pragma omp parallel
    {
        vector<double> A((size_t)k * k), b(k), v(k);
#pragma omp for schedule(dynamic, 16)
        for (int r = 0; r < m.n; ++r) {
            const int64_t lo = m.ptr[r], hi = m.ptr[r + 1];
            float* x = &X[(size_t)r * k];
            if (hi == lo) {
                std::fill(x, x + k, 0.f);
                continue;
            }
            std::fill(A.begin(), A.end(), 0.0);
            std::fill(b.begin(), b.end(), 0.0);
            for (int64_t p = lo; p < hi; ++p) {
                const float* f = &F[(size_t)m.idx[p] * k];
                const double y = (double)m.val[p] - mu;
                for (int a = 0; a < k; ++a) {
                    v[a] = f[a];
                    b[a] += y * v[a];
                    for (int c = 0; c <= a; ++c) A[a * k + c] += v[a] * v[c];  // lower triangle only
                }
            }
            const double reg = lambda * (double)(hi - lo);  // weighted-lambda regularisation
            for (int a = 0; a < k; ++a) A[a * k + a] += reg;
            if (cholesky_solve(A.data(), b.data(), k))
                for (int a = 0; a < k; ++a) x[a] = (float)b[a];
            else
                std::fill(x, x + k, 0.f);
        }
    }
}

static double rmse(const vector<int>& u, const vector<int>& i, const vector<float>& r, const vector<float>& U,
                   const vector<float>& V, int k, double mu, double lo, double hi) {
    if (r.empty()) return std::nan("");
    double se = 0;
#pragma omp parallel for reduction(+ : se) schedule(static)
    for (long t = 0; t < (long)r.size(); ++t) {
        const float* a = &U[(size_t)u[t] * k];
        const float* b = &V[(size_t)i[t] * k];
        double p = mu;
        for (int f = 0; f < k; ++f) p += (double)a[f] * b[f];
        p = std::min(hi, std::max(lo, p));
        const double e = p - r[t];
        se += e * e;
    }
    return std::sqrt(se / r.size());
}

// ----------------------------------------------------------------------------
int main(int argc, char** argv) {
    Config cfg;
    for (int a = 1; a < argc; ++a) {
        std::string k = argv[a];
        auto next = [&]() -> const char* {
            if (a + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", k.c_str());
                std::exit(1);
            }
            return argv[++a];
        };
        if (k == "--ratings") cfg.ratings = next();
        else if (k == "--out") cfg.out = next();
        else if (k == "--threads") cfg.threads = std::atoi(next());
        else if (k == "--rank") cfg.rank = std::atoi(next());
        else if (k == "--iters") cfg.iters = std::atoi(next());
        else if (k == "--lambda") cfg.lambda = std::atof(next());
        else if (k == "--topk") cfg.topk = std::atoi(next());
        else if (k == "--min-co") cfg.min_co = std::atoi(next());
        else if (k == "--shrink") cfg.shrink = std::atof(next());
        else if (k == "--holdout") cfg.holdout = std::atof(next());
        else if (k == "--seed") cfg.seed = std::strtoull(next(), nullptr, 10);
        else {
            std::fprintf(stderr,
                         "usage: recsys --ratings FILE [--out DIR] [--threads N] [--rank K] [--iters N]\n"
                         "              [--lambda L] [--topk K] [--min-co N] [--shrink S] [--holdout F] [--seed S]\n");
            return 1;
        }
    }
    if (cfg.ratings.empty()) {
        std::fprintf(stderr, "--ratings is required\n");
        return 1;
    }
    if (cfg.threads > 0) omp_set_num_threads(cfg.threads);
    const int threads = omp_get_max_threads();
    const int k = cfg.rank;

    // ---- Stage 1: load ------------------------------------------------------
    double t0 = omp_get_wtime();
    Raw raw;
    if (!load_ratings(cfg.ratings, raw)) {
        std::fprintf(stderr, "failed to read ratings from %s\n", cfg.ratings.c_str());
        return 1;
    }
    vector<int> du, di, user_ids, item_ids;
    densify(raw.uid, du, user_ids);
    densify(raw.iid, di, item_ids);
    const int nU = (int)user_ids.size(), nI = (int)item_ids.size();

    vector<int> tr_u, tr_i, te_u, te_i;
    vector<float> tr_r, te_r;
    float rmin = 1e9f, rmax = -1e9f;
    for (size_t t = 0; t < raw.r.size(); ++t) {
        rmin = std::min(rmin, raw.r[t]);
        rmax = std::max(rmax, raw.r[t]);
        const bool test = cfg.holdout > 0 && u01(cfg.seed * 1000003ULL + t) < cfg.holdout;
        auto& U = test ? te_u : tr_u;
        auto& I = test ? te_i : tr_i;
        auto& R = test ? te_r : tr_r;
        U.push_back(du[t]);
        I.push_back(di[t]);
        R.push_back(raw.r[t]);
    }
    double mu = 0;
    for (float x : tr_r) mu += x;
    mu /= std::max<size_t>(1, tr_r.size());

    CSR byUserRaw = build_csr(nU, tr_u, tr_i, tr_r);
    CSR byItemRaw = build_csr(nI, tr_i, tr_u, tr_r);

    // user-mean centred copies for adjusted cosine
    vector<double> umean(nU, 0.0);
    for (int u = 0; u < nU; ++u) {
        const int64_t lo = byUserRaw.ptr[u], hi = byUserRaw.ptr[u + 1];
        double s = 0;
        for (int64_t p = lo; p < hi; ++p) s += byUserRaw.val[p];
        umean[u] = hi > lo ? s / (hi - lo) : 0.0;
    }
    CSR byUserC = byUserRaw, byItemC = byItemRaw;
    for (int u = 0; u < nU; ++u)
        for (int64_t p = byUserC.ptr[u]; p < byUserC.ptr[u + 1]; ++p) byUserC.val[p] -= (float)umean[u];
    for (int i = 0; i < nI; ++i)
        for (int64_t p = byItemC.ptr[i]; p < byItemC.ptr[i + 1]; ++p) byItemC.val[p] -= (float)umean[byItemC.idx[p]];
    const double t_load = omp_get_wtime() - t0;

    std::fprintf(stderr, "[load] users=%d items=%d train=%zu test=%zu mu=%.4f threads=%d (%.3fs)\n", nU, nI,
                 tr_r.size(), te_r.size(), mu, threads, t_load);

    // ---- Stage 2: similarity --------------------------------------------------
    vector<int> sim_idx;
    vector<float> sim_val;
    double pairs = 0;
    t0 = omp_get_wtime();
    item_similarity(byUserC, byItemC, cfg, sim_idx, sim_val, pairs);
    const double t_sim = omp_get_wtime() - t0;
    std::fprintf(stderr, "[similarity] %.3fs  (%.2fM co-rating updates)\n", t_sim, pairs / 1e6);

    // ---- Stage 3: ALS -----------------------------------------------------------
    vector<float> U((size_t)nU * k, 0.f), V((size_t)nI * k);
    for (int i = 0; i < nI; ++i)  // thread-count independent initialisation
        for (int f = 0; f < k; ++f)
            V[(size_t)i * k + f] = (float)((u01(cfg.seed + (uint64_t)i * 7919 + f * 104729ULL) - 0.5) * 0.2);

    double t_als = 0;
    vector<double> tr_hist, te_hist;
    for (int it = 0; it < cfg.iters; ++it) {
        t0 = omp_get_wtime();
        als_half_step(byUserRaw, V, U, k, cfg.lambda, mu);
        als_half_step(byItemRaw, U, V, k, cfg.lambda, mu);
        t_als += omp_get_wtime() - t0;  // RMSE evaluation below is excluded from the timing
        const double a = rmse(tr_u, tr_i, tr_r, U, V, k, mu, rmin, rmax);
        const double b = rmse(te_u, te_i, te_r, U, V, k, mu, rmin, rmax);
        tr_hist.push_back(a);
        te_hist.push_back(b);
        std::fprintf(stderr, "[als] iter %2d  train RMSE %.4f  test RMSE %.4f\n", it + 1, a, b);
    }
    std::fprintf(stderr, "[als] %.3fs total\n", t_als);

    double base = std::nan("");
    if (!te_r.empty()) {
        double se = 0;
        for (float x : te_r) se += (x - mu) * (x - mu);
        base = std::sqrt(se / te_r.size());
    }

    // ---- Stage 4: artifacts -------------------------------------------------------
    std::string d = cfg.out + "/";
    write_bin(d + "user_ids.i32", user_ids.data(), user_ids.size() * 4);
    write_bin(d + "item_ids.i32", item_ids.data(), item_ids.size() * 4);
    write_bin(d + "sim_idx.i32", sim_idx.data(), sim_idx.size() * 4);
    write_bin(d + "sim_val.f32", sim_val.data(), sim_val.size() * 4);
    write_bin(d + "user_factors.f32", U.data(), U.size() * 4);
    write_bin(d + "item_factors.f32", V.data(), V.size() * 4);

    FILE* m = std::fopen((d + "meta.json").c_str(), "w");
    if (!m) {
        std::fprintf(stderr, "cannot write meta.json\n");
        return 2;
    }
    auto jnum = [&](double x) { return std::isnan(x) ? std::string("null") : std::to_string(x); };
    auto jarr = [&](const vector<double>& v) {
        std::fprintf(m, "[");
        for (size_t t = 0; t < v.size(); ++t) std::fprintf(m, "%s%s", t ? "," : "", jnum(v[t]).c_str());
        std::fprintf(m, "]");
    };
    std::fprintf(m, "{\n");
    std::fprintf(m, "  \"threads\": %d,\n  \"n_users\": %d,\n  \"n_items\": %d,\n", threads, nU, nI);
    std::fprintf(m, "  \"n_train\": %zu,\n  \"n_test\": %zu,\n", tr_r.size(), te_r.size());
    std::fprintf(m, "  \"global_mean\": %.8f,\n  \"rating_min\": %.2f,\n  \"rating_max\": %.2f,\n", mu, rmin, rmax);
    std::fprintf(m, "  \"rank\": %d,\n  \"iters\": %d,\n  \"lambda\": %.6f,\n  \"topk\": %d,\n", k, cfg.iters,
                 cfg.lambda, cfg.topk);
    std::fprintf(m, "  \"min_co\": %d,\n  \"shrink\": %.3f,\n  \"holdout\": %.3f,\n", cfg.min_co, cfg.shrink,
                 cfg.holdout);
    std::fprintf(m, "  \"time_load_s\": %.6f,\n  \"time_similarity_s\": %.6f,\n  \"time_als_s\": %.6f,\n", t_load,
                 t_sim, t_als);
    std::fprintf(m, "  \"time_compute_s\": %.6f,\n", t_sim + t_als);
    std::fprintf(m, "  \"baseline_test_rmse\": %s,\n", jnum(base).c_str());
    std::fprintf(m, "  \"train_rmse\": ");
    jarr(tr_hist);
    std::fprintf(m, ",\n  \"test_rmse\": ");
    jarr(te_hist);
    std::fprintf(m, "\n}\n");
    std::fclose(m);
    std::fprintf(stderr, "[done] artifacts written to %s\n", cfg.out.c_str());
    return 0;
}
