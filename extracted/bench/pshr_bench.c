// pshr_bench.c - C port of the PS-HR binary associative memory hot loop.
// Dense scan vs event-driven cascade (prefix pulse -> gated full compare).
//
// (Normally built via `make build` from the project root.)
// Build (scalar-only, mimics a CPU with no SIMD):
//   gcc -O2 -mpopcnt -fno-tree-vectorize -fno-tree-slp-vectorize -o bench_scalar pshr_bench.c
// Build (auto-vectorized, AVX2/AVX512 where available):
//   gcc -O3 -march=native -o bench_simd pshr_bench.c
// Run: ./bench_scalar [N] [cold=0|1]

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define W   128          // 8192-bit vectors
#define W1  16           // prefix words (1024 bits) used as the cheap "pulse"
#define NQ  200

static uint64_t rs = 0x1234567ULL;
static inline uint64_t rnd(void) {   // splitmix64
    uint64_t z = (rs += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static double now_ns(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e9 + t.tv_nsec;
}
// noise masks with approximate flip probabilities
static inline uint64_t m_6(void)  { return rnd() & rnd() & rnd() & rnd(); }                       // ~6.25%
static inline uint64_t m_3(void)  { return rnd() & rnd() & rnd() & rnd() & rnd(); }                       // ~3.1%
static inline uint64_t m_12(void) { return rnd() & rnd() & rnd(); }                               // ~12.5%
static inline uint64_t m_18(void) { uint64_t a = m_12(); return a | (rnd() & rnd() & rnd() & rnd() & ~a); } // ~18%

static uint64_t *full, *pre, *rest;
static size_t N;

static int dense(const uint64_t *q) {
    int best = 0; uint32_t bh = 0xFFFFFFFF;
    for (size_t i = 0; i < N; i++) {
        const uint64_t *p = full + i * W; uint32_t h = 0;
        for (int w = 0; w < W; w++) h += __builtin_popcountll(p[w] ^ q[w]);
        if (h < bh) { bh = h; best = (int)i; }
    }
    return best;
}

static uint32_t *h1buf;
static int COLD = 0; static volatile uint64_t sink; static uint64_t *flushbuf; static size_t FLUSHW = 8u<<20; // 64 MB
static void flush(void){ if(!COLD) return; uint64_t s=0; for(size_t i=0;i<FLUSHW;i+=8) s+=flushbuf[i]; sink=s; }
static int cascade(const uint64_t *q, int margin, double *active_frac) {
    uint32_t mn = 0xFFFFFFFF;
    for (size_t i = 0; i < N; i++) {
        const uint64_t *p = pre + i * W1; uint32_t h = 0;
        for (int w = 0; w < W1; w++) h += __builtin_popcountll(p[w] ^ q[w]);
        h1buf[i] = h; if (h < mn) mn = h;
    }
    uint32_t thr = mn + margin, bh = 0xFFFFFFFF; int best = 0; size_t act = 0;
    for (size_t i = 0; i < N; i++) {
        if (h1buf[i] > thr) continue;
        act++;
        const uint64_t *p = rest + i * (W - W1); uint32_t h = h1buf[i];
        for (int w = 0; w < W - W1; w++) h += __builtin_popcountll(p[w] ^ q[W1 + w]);
        if (h < bh) { bh = h; best = (int)i; }
    }
    *active_frac = (double)act / N;
    return best;
}

static void run(const char *label, int cluster_size, uint64_t (*cn)(void)) {
    // build store
    uint64_t base[W];
    for (size_t i = 0; i < N; i++) {
        if (cluster_size <= 1) { for (int w = 0; w < W; w++) full[i * W + w] = rnd(); }
        else {
            if (i % cluster_size == 0) for (int w = 0; w < W; w++) base[w] = rnd();
            for (int w = 0; w < W; w++) full[i * W + w] = base[w] ^ cn();
        }
        memcpy(pre + i * W1, full + i * W, W1 * 8);
        memcpy(rest + i * (W - W1), full + i * W + W1, (W - W1) * 8);
    }
    static uint64_t Q[NQ][W]; int tgt[NQ];
    for (int k = 0; k < NQ; k++) {
        tgt[k] = (int)(rnd() % N);
        for (int w = 0; w < W; w++) Q[k][w] = full[(size_t)tgt[k] * W + w] ^ m_18();
    }
    int dref[NQ]; int ok = 0;
    double t0 = now_ns();
    double tdsum=0; for (int k = 0; k < NQ; k++) { flush(); double a0=now_ns(); dref[k] = dense(Q[k]); tdsum+=now_ns()-a0; ok += dref[k] == tgt[k]; }
    double td = COLD ? tdsum / NQ : (now_ns() - t0) / NQ;
    printf("| %s | dense | %.3f | %.0f%% | 100%% | 1.000 | 100%% |\n", label, td / 1e6, 100.0 * ok / NQ);
    #ifndef MARGINS
#define MARGINS {0, 30, 60, 120, 240}
#endif
    int margins[] = MARGINS; int NM = sizeof(margins)/sizeof(int);
    for (int mi = 0; mi < NM; mi++) {
        int hit = 0, agree = 0; double af = 0, a;
        double tsum = 0;
        for (int k = 0; k < NQ; k++) {
            flush(); double a0 = now_ns();
            int b = cascade(Q[k], margins[mi], &a); tsum += now_ns() - a0; af += a;
            hit += b == tgt[k]; agree += b == dref[k];
        }
        double t = tsum / NQ; af /= NQ;
        double ops = (double)W1 / W + af * (double)(W - W1) / W;
        printf("| %s | cascade m=%d | %.3f | %.0f%% | %.1f%% | %.3f | %.0f%% |\n",
               label, margins[mi], t / 1e6, 100.0 * hit / NQ, af * 100, ops, 100.0 * agree / NQ);
        (void)td;
    }
}

int main(int argc, char **argv) {
    N = argc > 1 ? (size_t)atol(argv[1]) : 100000;
    full = aligned_alloc(64, N * W * 8);
    pre  = aligned_alloc(64, N * W1 * 8);
    rest = aligned_alloc(64, N * (W - W1) * 8);
    h1buf = malloc(N * 4);
    COLD = argc > 2 && atoi(argv[2]);
    flushbuf = aligned_alloc(64, FLUSHW * 8); memset(flushbuf, 1, FLUSHW * 8);
    printf("N=%zu, 8192-bit vectors, query noise ~18%%, cache=%s\n\n", N, COLD ? "cold (64MB flushed before every query)" : "warm");
    printf("| store | mode | ms/query | recall(planted) | nodes fired | ops fraction | agrees w/ dense |\n|---|---|---|---|---|---|---|\n");
    run("uniform", 1, NULL);
    run("clusters of 500, 3% intra-noise (near-duplicates)", 500, m_3);
    run("clusters of 50, 6% intra-noise", 50, m_6);
    run("clusters of 50, 12% intra-noise", 50, m_12);
    return 0;
}
