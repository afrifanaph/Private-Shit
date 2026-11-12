# PS-HR Prototype: Measured Results

Data: SYNTHETIC 40-intent confusable benchmark (pseudo-words, 12% typo rate, 25% distractor words); not a real-language benchmark. Hardware: x86_64, 1 thread, Python 3.12.3, NumPy 2.5.4.

All numbers are produced by `experiments/run_experiments.py`; rerun to reproduce. Single-threaded NumPy; latencies include Python overhead and are NOT representative of optimized C on an edge CPU.

## 1. Accuracy vs training shots per class (mean ± std over seeds)

| Model | 5 shots | 20 shots | 80 shots |
|---|---|---|---|
| BinaryHDC-8192b-noretrain | 68.0 ± 2.2 | 84.8 ± 1.0 | 89.2 ± 0.5 |
| BinaryHDC-8192b-retrain | 68.0 ± 2.2 | 86.1 ± 1.1 | 91.1 ± 0.1 |
| PhaseHDC-Q16-8192b-retrain | 69.5 ± 2.7 | 86.6 ± 0.6 | 91.3 ± 0.3 |
| LogReg-hash-f32(2^14x2) | 86.8 ± 1.5 | 93.0 ± 0.5 | 94.5 ± 0.0 |
| LogReg-hash-int8(2^14x2) | 86.8 ± 1.5 | 93.0 ± 0.4 | 94.5 ± 0.0 |
| LogReg-hash-f32(2^10x2) | 85.4 ± 1.8 | 92.6 ± 0.4 | 93.5 ± 0.0 |
| TF-IDF nearest-centroid(hash 2^14x2) | 83.7 ± 1.9 | 92.2 ± 0.2 | 93.2 ± 0.0 |

## 2. Model size and per-query latency (80 shots)

| Model | Model bytes | Latency (µs/query, incl. encoding) |
|---|---|---|
| BinaryHDC | 40,960 | 294 |
| PhaseHDC-Q16 | 40,960 | 985 |
| LogReg-hash-f32(2^14x2) | 5,242,880 | 1677 |
| LogReg-hash-int8(2^14x2) | 1,310,880 | 2495 |
| LogReg-hash-f32(2^10x2) | 327,680 | 644 |

## 3. H1: procedural item memory

- Distinct features seen: 5,120
- Stored item-memory table would cost: 5,242,880 bytes; procedural: 0 bytes (seed = hash of the token string)
- Regeneration latency: 8.5 µs/token vs dict lookup 0.05 µs/token
- Caveat: only the random item vectors are procedural. The class prototypes are learned and are stored.

## 4. Noise robustness (accuracy when a fraction of query components is randomized)

| Model | p=0.0 | p=0.1 | p=0.2 | p=0.3 | p=0.4 | p=0.5 |
|---|---|---|---|---|---|---|
| BinaryHDC | 90.9 | 91.1 | 90.8 | 90.3 | 89.6 | 89.2 |
| PhaseHDC-Q16 | 91.2 | 91.5 | 90.8 | 90.4 | 90.0 | 89.8 |

## 5. Holographic key-value capacity (retrieval accuracy at equal bit budget)

**binary**

| bits | k=8 | k=16 | k=32 | k=64 | k=128 | k=256 |
|---|---|---|---|---|---|---|
| 1024 | 100 | 100 | 95 | 53 | 23 | 8 |
| 2048 | 100 | 100 | 100 | 94 | 59 | 22 |
| 4096 | 100 | 100 | 100 | 100 | 90 | 59 |
| 8192 | 100 | 100 | 100 | 100 | 100 | 92 |

**phase**

| bits | k=8 | k=16 | k=32 | k=64 | k=128 | k=256 |
|---|---|---|---|---|---|---|
| 1024 | 100 | 100 | 71 | 27 | 9 | 5 |
| 2048 | 100 | 100 | 97 | 73 | 32 | 11 |
| 4096 | 100 | 100 | 100 | 98 | 66 | 30 |
| 8192 | 100 | 100 | 100 | 100 | 97 | 67 |

## 6. H3: event-driven cascade, large associative store (20% query noise, 8192-bit vectors)

| N items | mode | ms/query | recall | ops fraction | nodes activated |
|---|---|---|---|---|---|
| 1,000 | dense | 0.22 | 100% | 1.000 | 100.0% |
| 1,000 | cascade_m0 | 0.06 | 100% | 0.126 | 0.1% |
| 1,000 | cascade_m40 | 0.05 | 100% | 0.126 | 0.1% |
| 1,000 | cascade_m80 | 0.05 | 100% | 0.126 | 0.1% |
| 1,000 | cascade_m160 | 0.05 | 100% | 0.126 | 0.1% |
| 10,000 | dense | 2.98 | 100% | 1.000 | 100.0% |
| 10,000 | cascade_m0 | 0.66 | 100% | 0.125 | 0.0% |
| 10,000 | cascade_m40 | 0.64 | 100% | 0.125 | 0.0% |
| 10,000 | cascade_m80 | 0.64 | 100% | 0.125 | 0.0% |
| 10,000 | cascade_m160 | 0.64 | 100% | 0.125 | 0.0% |
| 100,000 | dense | 48.10 | 100% | 1.000 | 100.0% |
| 100,000 | cascade_m0 | 6.32 | 100% | 0.125 | 0.0% |
| 100,000 | cascade_m40 | 6.09 | 100% | 0.125 | 0.0% |
| 100,000 | cascade_m80 | 9.12 | 100% | 0.125 | 0.0% |
| 100,000 | cascade_m160 | 6.53 | 100% | 0.125 | 0.0% |

## 7. Cascade on the class prototypes of the intent task

| mode | accuracy | ops fraction |
|---|---|---|
| dense | 90.9% | 1.00 |
| m0 | 88.1% | 0.15 |
| m20 | 90.7% | 0.16 |
| m40 | 90.9% | 0.18 |
| m80 | 90.9% | 0.30 |
| m160 | 90.9% | 0.82 |
## 8. C port: dense vs cascade on the associative store (N=100,000, 8192-bit vectors, 18% query noise)

Hardware: Intel Xeon @ 2.10 GHz (one core), gcc 13.3. Source: `bench/pshr_bench.c`. Raw tables: `results/c_results_*.md`.

| Build | Cache | Dense ms/query | Cascade ms/query (low activity) | Speedup |
|---|---|---|---|---|
| scalar (-O2 -mpopcnt, no vectorization) | warm (repeated queries) | 19.3 | 1.1 | ~17x (inflated by cache residency of the 12.8 MB prefix block) |
| scalar | cold (64 MB flushed per query) | 19.1 | 1.6-2.1 | ~9-12x |
| SIMD (-O3 -march=native, AVX-512) | warm | 8.3 | 0.76 | ~10x |
| SIMD | cold | 7.9 | 1.1-1.4 | ~6-7x |

The cold-cache scalar figure is the honest one for a device whose cache is smaller than the store; the ideal operation-count bound is 8x (prefix = 12.5% of the work).

**Activity sweep (uniform store, cold cache, margin forced up so most nodes fire):**

| Nodes fired | Ops fraction | Scalar cascade / dense | SIMD cascade / dense |
|---|---|---|---|
| ~0% | 0.125 | ~0.09-0.11 | ~0.15 |
| 72.6% | 0.760 | 0.93 | 1.06 |
| 94.0% | 0.947 | 1.04 | 1.16 |
| 99-100% | 1.00 | ~0.93-1.0 | ~1.12 |

Cascade stops paying off between roughly 70% and 95% activity on this CPU; with SIMD dense code it breaks even earlier (about 70%). Near-duplicate clusters (3% intra-cluster noise, 500 per cluster) still fired only 0.5% of nodes, so the planted-recall tests here remain easy; a store with many near-ties would fire more.

## 9. Hard-case sweep: near-tie stores, prefix length, adaptive margin (C, N=100,000, cold cache)

Source: `bench/pshr_sweep.c`; full tables in `results/sweep_results.md` and `results/sweep_adaptive.md`. Stores are built from clusters of near-duplicates (members = shared base vector with 1.6-3.1% bit noise) so that many nodes genuinely compete; queries have 12.5-25% bit noise. Each row uses 100 queries, so a "100%" recall figure has a lower confidence bound of roughly 96-97%.

**Findings**

1. **Speed is bounded by the number of true near-ties.** Measured speedups follow `1 / (f_prefix + a * (1 - f_prefix))`. With 20,000-item clusters (20% of the store competing), scalar speedup falls from ~10x to ~3.5-4x, and SIMD from ~7x to ~2.3-2.8x. The cascade cannot beat the count of items that are genuinely close.
2. **The margin must exceed prefix-distance noise or recall collapses.** Margin 0 gave 95% recall on 3.1% clusters and 63% on 1.6% clusters (1024-bit prefix); 256-bit prefix at margin 0 gave 19%. Any margin of roughly 20+ bits restored 100% recall without increasing activity beyond the cluster itself.
3. **Prefix length trade-off.** 256-512 bits: fast when it works, but the safe window is narrow (a slightly larger margin fires the whole store). 4096 bits: safe but capped near 2x. **1024-2048 bits is the practical range.**
4. **Adaptive margin works without per-store tuning.** Setting margin = z * sqrt(n p (1-p)), with p estimated from the best prefix distance, gave 100% recall at z = 1, 2 and 3 on all seven store types tested, and z = 1 was fastest. Recommended default: **z = 2** for headroom. Implemented in `SplitStore.cascade(q)` (Python) and `cascade()` with negative margin (C).
5. **When everything fires, the cascade costs ~10-20% extra** (SIMD build: 0.81-0.90x of dense).

**Caveats.** Noise is i.i.d. bit flips, which is the model the sigma rule assumes. Hypervectors from a real encoder will have structured, non-i.i.d. distance distributions, so the z rule needs validation on encoder output before it is trusted. Single x86 core; ARM and energy not measured.


## 10. Semantic-encoder experiments (negative results)

Full output: `results/semantic.md`; reproduce with `make experiments-semantic`. On a 16-intent hand-written few-shot paraphrase benchmark (48 train / 128 test, SE about 3-4 points):
lexical HDC 63.3%, lexical TF-IDF logistic regression 71.1%, random-indexing semantic-only HDC 57.0%, hybrid 62.5-65.6%, float RI centroid 64.8%.
Random indexing from a 4.4M-word local docstring corpus therefore did not help reliably; binarizing the embeddings cost ~8 points versus floats; IDF-weighted bundling was worse than unweighted at every setting tried (best 61.7%). The corpus is API documentation, not conversational language, so this tests the data more than the method.

## 11. Sharded holographic knowledge memory (Phase D)

Full output: `results/knowledge.md`; reproduce with `make experiments-knowledge`. 5,000 entities, 16 relations, 20,000 facts.

| Result | Value |
|---|---|
| 1-hop accuracy, D=8192, 32 facts/shard (256 bits/fact) | 100.0% |
| 1-hop accuracy, D=8192, 128 facts/shard (64 bits/fact) | 98.5% |
| 1-hop accuracy, D=16384, 256 facts/shard (64 bits/fact) | 99.5% |
| Multi-hop (D=8192, 32 facts/shard): 1 / 2 / 3 / 4 hops | 100.0 / 100.0 / 99.9 / 99.8% |
| Accuracy with 10% / 20% / 30% of stored bits flipped (1-hop) | 100.0 / 99.6 / 80.5% |
| Cascade cleanup (2048-bit prefix, z=2): accuracy, compare-ops vs dense | 99.9%, 0.25x |
| Explicit-table lower bound | 17 bits/fact (holographic: 64-1024 bits/fact) |

Interpretation: associative multi-step reasoning over symbols works and degrades gracefully under memory corruption, but holographic storage is **not** compressive for arbitrary facts (consistent with the proposal's own caveat that procedural methods cannot generate missing information). Caveats: synthetic random graph with functional relations; no text-to-triple extraction; single x86 core; Python timings.


## 12. Compositional generalization (H5) and margin training

Full output: `results/compositional.md`. 10 random splits; 15 of 25 action+object pairs seen in training; score = both parts correct.

| Test set | flat HDC | factored HDC (plain) | factored HDC (margin 0.12) | factored LogReg |
|---|---|---|---|---|
| held-out pairs, known words | 0.0% | 45.2% | 89.2% | 94.6% |
| held-out pairs, one new synonym | 0.0% | 6.8% | 13.8% | 10.6% |
| seen pairs, known words | 51.0% | 80.3% | 98.2% | 98.7% |

The factored model stores 10 prototypes instead of 25. Plain prototype bundling soaks up confounders (each object appears with only 3 of 5 actions), so margin training (keep updating until the right class wins by a gap) is what makes it work. The margin was chosen on this task; on the 16-intent paraphrase benchmark the same change gave 63% -> 67-69% at margins 0.12-0.16 but 55% at 0.04.

## 13. End-to-end text QA

Full output: `results/qa.md`. Statements/questions in a synthetic world (60 people, 20 cities, 6 countries, 10 companies, 6 relations); classifiers trained on 3 phrasings per type.

| | Trained phrasings | Unseen phrasings |
|---|---|---|
| Reading (statement -> relation) | 100% | 74% |
| Question type | 100% | 48% |
| End-to-end 1-step / 2-step | 99.8% / 96.5% | 26% / 10% |
| Memory alone, perfect reading (1-step / 2-step) | 99.8% / 96.5% | 99.5% / 97.3% |

With 0/1/2 extra training phrasings per question type (fixed test phrasing, 10 worlds): question type 27% / 50% / 76%; end-to-end 1-step 0% / 27% / 46%; statement reading was non-monotonic (89% / 70% / 79%), so this part is noisy. Interpretation: the holographic memory is not the bottleneck; matching words rather than meaning is.


## 14. Sentence-embedding hook (plumbing verified, real model untested)

`src/embed_hdc.py`: embedding -> centre -> sign random projection -> bits (Hamming distance tracks angle; unit-tested), plus a hybrid lexical+embedding encoder usable by every classifier and by `TextQA`. Output with a weak stand-in embedding is in `results/embedding_standin.md`: hybrid + margin training 71.1% on the paraphrase benchmark vs 63.3% plain lexical and 71.1% logistic regression (128 test items, SE ~3-4 points, so indicative only). Whether a real pretrained sentence model closes the gap has not been measured.
