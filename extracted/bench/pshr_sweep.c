// pshr_sweep.c - hard-case sweep for the event-driven cascade.
// Usage: ./sweep N cluster_size k_intra k_query w1 [cold=1]
//   k_intra : cluster members = base ^ noise with flip prob 2^-k_intra (4=6.25%, 5=3.1%, 6=1.6%)
//   k_query : query = item ^ noise with flip prob 2^-k_query            (2=25%, 3=12.5%)
//   w1      : prefix length in 64-bit words (of 128)
// Big clusters + small intra-noise => many near-ties => high node activation.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define W 128
#define NQ 100
static uint64_t rs = 0xC0FFEEULL;
static inline uint64_t rnd(void) {
    uint64_t z = (rs += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static double now_ns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e9 + t.tv_nsec; }
static inline uint64_t noise(int k) { uint64_t m = rnd(); for (int i = 1; i < k; i++) m &= rnd(); return m; }

static uint64_t *full, *pre, *rest, *flushbuf; static uint32_t *h1buf;
static size_t N; static int w1, COLD; static volatile uint64_t sink;
static const size_t FLUSHW = 8u << 20;
static void flush(void) { if (!COLD) return; uint64_t s = 0; for (size_t i = 0; i < FLUSHW; i += 8) s += flushbuf[i]; sink = s; }

static int dense(const uint64_t *q) {
    int best = 0; uint32_t bh = ~0u;
    for (size_t i = 0; i < N; i++) {
        const uint64_t *p = full + i * W; uint32_t h = 0;
        for (int w = 0; w < W; w++) h += __builtin_popcountll(p[w] ^ q[w]);
        if (h < bh) { bh = h; best = (int)i; }
    }
    return best;
}
static int cascade(const uint64_t *q, int margin, double *act) {
    uint32_t mn = ~0u;
    for (size_t i = 0; i < N; i++) {
        const uint64_t *p = pre + i * w1; uint32_t h = 0;
        for (int w = 0; w < w1; w++) h += __builtin_popcountll(p[w] ^ q[w]);
        h1buf[i] = h; if (h < mn) mn = h;
    }
    // margin >= 0: fixed bits. margin < 0: adaptive, z = -margin sigmas, sigma estimated from the best prefix distance
    double nb = w1 * 64.0, ph = mn / nb;
    uint32_t thr = margin >= 0 ? mn + margin : mn + (uint32_t)ceil(-margin * sqrt(nb * ph * (1 - ph)));
    uint32_t bh = ~0u; int best = 0; size_t a = 0;
    for (size_t i = 0; i < N; i++) {
        if (h1buf[i] > thr) continue;
        a++;
        const uint64_t *p = rest + i * (W - w1); uint32_t h = h1buf[i];
        for (int w = 0; w < W - w1; w++) h += __builtin_popcountll(p[w] ^ q[w1 + w]);
        if (h < bh) { bh = h; best = (int)i; }
    }
    *act = (double)a / N; return best;
}

int main(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage: %s N cluster k_intra k_query w1 [cold]\n", argv[0]); return 1; }
    N = atol(argv[1]); int cl = atoi(argv[2]), ki = atoi(argv[3]), kq = atoi(argv[4]); w1 = atoi(argv[5]);
    COLD = argc > 6 ? atoi(argv[6]) : 1;
    full = aligned_alloc(64, N * W * 8); pre = aligned_alloc(64, N * w1 * 8);
    rest = aligned_alloc(64, N * (W - w1) * 8); h1buf = malloc(N * 4);
    flushbuf = aligned_alloc(64, FLUSHW * 8); memset(flushbuf, 1, FLUSHW * 8);
    uint64_t base[W];
    for (size_t i = 0; i < N; i++) {
        if (i % cl == 0) for (int w = 0; w < W; w++) base[w] = rnd();
        for (int w = 0; w < W; w++) full[i * W + w] = base[w] ^ (cl > 1 ? noise(ki) : 0);
        memcpy(pre + i * w1, full + i * W, w1 * 8);
        memcpy(rest + i * (W - w1), full + i * W + w1, (W - w1) * 8);
    }
    static uint64_t Q[NQ][W]; int tgt[NQ], dref[NQ]; int okd = 0; double td = 0;
    for (int k = 0; k < NQ; k++) {
        tgt[k] = (int)(rnd() % N);
        for (int w = 0; w < W; w++) Q[k][w] = full[(size_t)tgt[k] * W + w] ^ noise(kq);
    }
    for (int k = 0; k < NQ; k++) { flush(); double t0 = now_ns(); dref[k] = dense(Q[k]); td += now_ns() - t0; okd += dref[k] == tgt[k]; }
    td /= NQ;
    printf("| %zu | %d | %.1f%% | %.1f%% | %d | dense | - | %.2f | %d%% | 100%% | 1.00 | 100%% |\n",
           N, cl, 100.0 / (1 << ki), 100.0 / (1 << kq), w1 * 64, td / 1e6, okd);
    int margins[] = {0, 20, 40, 80, 160, 320, -1, -2, -3}; int NM = 9;
    for (int mi = 0; mi < NM; mi++) {
        double ts = 0, af = 0, a; int hit = 0, ag = 0;
        for (int k = 0; k < NQ; k++) {
            flush(); double t0 = now_ns(); int b = cascade(Q[k], margins[mi], &a); ts += now_ns() - t0;
            af += a; hit += b == tgt[k]; ag += b == dref[k];
        }
        ts /= NQ; af /= NQ;
        char mlab[32]; if (margins[mi] >= 0) snprintf(mlab, 32, "%d", margins[mi]); else snprintf(mlab, 32, "auto z=%d", -margins[mi]);
        printf("| %zu | %d | %.1f%% | %.1f%% | %d | cascade | %s | %.2f | %d%% | %.1f%% | %.2f | %d%% |\n",
               N, cl, 100.0 / (1 << ki), 100.0 / (1 << kq), w1 * 64, mlab, ts / 1e6, hit,
               af * 100, td / ts, ag);
    }
    return 0;
}
