# PS-HR: Phase-Shift Pulse and Holographic-Resonant prototype

A reference implementation and validation suite for the PS-HR research proposal: sparse, procedural,
event-driven associative computation on resource-constrained CPUs, built on hyperdimensional computing (HDC).

**Status: working software prototype. Not a language model.** It classifies short text into a fixed label set,
stores and retrieves bundled key-value records, and searches large binary associative stores with an early-exit
cascade. See "Results" and "Limitations" before drawing conclusions.

## Layout

```
pshr/
  README.md
  Makefile                   build / test / benchmark entry points (run `make help`)
  requirements.txt
  src/pshr.py                library: procedural item memory, Binary HDC, Phase HDC, classifier, cascade search
  src/knowledge.py           sharded holographic triple store with multi-hop queries (Phase D)
  src/embed_hdc.py           plug a sentence-embedding model into the HDC machinery (sign random projection + hybrid encoder)
  src/qa.py                  tiny text QA: read sentences -> store facts -> answer 1- and 2-step questions
  tests/test_pshr.py         17 sanity tests (seconds)
  experiments/
    run_experiments.py       Python suite: accuracy vs baselines, size/latency, H1 storage, noise, capacity, cascade
    run_knowledge.py         knowledge-memory suite: bits/fact, multi-hop, corruption tolerance, cascade cleanup
    run_semantic.py          negative-result experiments: random-indexing semantics, IDF weighting
    run_compositional.py     H5: unseen action+object combinations, flat vs factored, margin training
    run_qa.py                end-to-end text QA evaluation incl. unseen phrasings
    run_embedding.py         lexical vs embedding vs hybrid encoders; `--model ri` (stand-in) or a sentence-transformers name (UNTESTED)
    intents_sysadmin.py      small hand-written few-shot paraphrase benchmark (16 intents, 48 train / 128 test)
  bench/
    pshr_bench.c             C dense-vs-cascade benchmark (warm/cold cache, activity sweep)
    pshr_sweep.c             C hard-case sweep: near-tie clusters, prefix length, fixed vs adaptive margin
  scripts/export_datasets.py Banking77 / CLINC150 -> TSV (needs network; untested, see file header)
  scripts/build_corpus.py    builds a local English corpus from installed Python docstrings (for run_semantic.py)
  results/                   measured outputs from the build machine (see below)
```

## Quick start

```bash
pip install -r requirements.txt          # NumPy >= 2.0 is required (np.bitwise_count)
make test                                # 17 tests, ~1 s
make experiments                         # Python suite on synthetic data, ~3-4 min -> results/REPORT.md
make bench                               # C benchmark, scalar + SIMD builds (needs gcc; ~1 min)
make sweep                               # C near-tie sweep
make experiments-knowledge               # knowledge-memory suite (~1-2 min) -> results/knowledge.md
make experiments-semantic                # semantic negative results (~1 min) -> results/semantic.md-style output
make experiments-compositional           # H5 (~1 min) -> results/compositional.md
make experiments-qa                      # text QA (~1-2 min) -> results/qa.md
```

Build outputs go to `build/` (override with `make build BUILD=/some/dir`). On non-x86 targets, edit the C flags in
the Makefile (`-mpopcnt` and `-march=native` are x86-oriented; try `-mcpu=native` on ARM).

### Run everything on GitHub (no local setup, includes the real-model and ARM tests)

1. Create a new **public** GitHub repository and upload this folder's contents (keep the `.github/` folder).
2. Open the **Actions** tab -> **PS-HR validation** -> **Run workflow**.
3. When it finishes, download the results from the **Artifacts** list at the bottom of the run page:
   `bench-ubuntu-latest` (x86 speed), `bench-ubuntu-24.04-arm` (ARM speed), `results-synthetic`, `results-real-model`
   (Banking77 and CLINC150 with `all-MiniLM-L6-v2`, lexical vs embedding vs hybrid, plus the paraphrase/QA tests).

The workflow (`.github/workflows/validate.yml`) was checked here only for valid YAML and by dry-running its commands locally;
it has **not** run on GitHub. The ARM runner is free for public repositories only. The `real-model` job is marked
`continue-on-error` so a failure there (e.g. a dataset field name in `scripts/export_datasets.py`) does not hide the other results.
Energy is not measured by any of this (GitHub runners have no power meter).

Caution when reading the hybrid results: `margin` is an absolute similarity gap and each extra encoder is run both plain and
margin-trained for that reason; adding an embedding code that carries no signal can *hurt* the hybrid (seen with a stand-in).

### Try a real sentence model (the experiment that matters most; untested in the build sandbox)

```bash
pip install sentence-transformers
cd experiments
python run_embedding.py --model sentence-transformers/all-MiniLM-L6-v2
```

It prints paraphrase-benchmark accuracy and QA accuracy on unseen phrasings for lexical, embedding-only and hybrid encoders.
Compare against `results/embedding_standin.md` (weak stand-in embedding) to see what a real model adds.

### Run on real data

```bash
pip install datasets
python scripts/export_datasets.py data/
make experiments-real DATA=data/banking77        # or data/clinc150
```

The TSV format is `label<TAB>text`, one example per line, so any intent dataset works.

## What is implemented

| Component | File | Notes |
|---|---|---|
| Procedural item memory | `src/pshr.py` | 64-bit seed (hash of token) -> hypervector via splitmix64; no stored table |
| Binary HDC | `src/pshr.py` | packed uint64, XOR bind, majority bundle, popcount similarity, perceptron-style retraining |
| Phase HDC (discrete FHRR, Q=16) | `src/pshr.py` | bind = add mod Q, bundle = phasor sum, LUT similarity, same bit budget as binary |
| Event-driven cascade | `src/pshr.py`, `bench/*.c` | cheap prefix "pulse" to all nodes; only nodes within an adaptive margin get the full comparison |
| Sharded holographic knowledge memory | `src/knowledge.py` | facts bound as `S ^ roll(R) ^ O`, bundled per hash-selected shard; query touches one shard; cleanup by dense or cascade search |
| Baselines | `experiments/run_experiments.py` | hashed TF-IDF logistic regression (float32 and int8), TF-IDF nearest centroid |

## Results (measured on the build machine: 1 core of an x86 Xeon @ 2.1 GHz, gcc 13, NumPy 2.5)

Full tables: `results/REPORT.md`. Raw data: `results/results.json`, `results/sweep_*.md`, `results/c_results_*.md`.

- **Accuracy (synthetic 40-intent benchmark):** HDC 91.1-91.3% vs logistic regression 94.5% at 80 shots per class;
  about 68-70% vs 87% at 5 shots. The model is 41 KB (vs 327 KB-5.2 MB for the baselines).
- **Cascade search, 100,000 x 8192-bit items, cold cache:** about 9-12x faster (scalar) and 6-7x (SIMD) at 100% recall when
  few nodes fire. Speedup follows `1 / (prefix_fraction + fraction_of_true_near_ties)`; with 20% near-ties it drops to ~3.5-4x
  (scalar) / ~2.5x (SIMD). If every node fires, the cascade is 10-20% slower than a dense scan.
- **Margin matters:** margin 0 loses recall on near-tie stores (63-95%). The adaptive rule
  `margin = z * sqrt(n p (1-p))` (z = 2 recommended) kept 100% recall on all seven store types tested.
- **Capacity (key-value records in one vector, 8192 bits):** ~100% up to 128 pairs, 92% at 256 (binary) vs 67% (phase).
- **Knowledge memory (20,000 facts, 5,000 entities, D=8192):** 100% 1-hop accuracy at 32 facts/shard (256 bits/fact) and 98.5% at
  128 facts/shard (64 bits/fact); 99.7-100% through 4 hops; still 100% with 10% of stored bits flipped (99.6% at 20%, 80% at 30%).
  An explicit table needs only ~17 bits/fact, so this is **4-15x larger, not smaller**: the benefit is robustness and cheap composition,
  not compression. Cascade cleanup over the entity memory cuts compare operations to ~25% at 99.9-100% accuracy.
- **Unseen combinations (H5):** a factored memory (5 action + 5 object prototypes instead of 25 pair prototypes) gets 89% on
  action+object pairs never seen in training with margin training (45% with plain training; logistic regression 95%). The margin value was
  chosen on this task, so 89% is optimistic. With a genuinely new synonym every model drops to 7-14%.
- **Text question answering (new):** the memory part is fine (99.5% one-step, 97.3% two-step questions with perfect reading), but on
  phrasings never seen in training the reading side collapses (statements 74%, question type 48%), giving 26% / 10% end-to-end.
  On trained phrasings it is ~100% / 96.5%. More example phrasings help (question type 27% -> 50% -> 76% with 0/1/2 extra).
- **Margin training:** `HDClassifier(..., margin=m)` keeps updating prototypes until the right class wins by a gap. Large help on H5
  (45% -> 89%), small and noisy on the paraphrase benchmark (63% -> 67-69%, but 55% at one setting). Needs per-task validation.
- **Embedding hook (new, plumbing only):** `src/embed_hdc.py` turns any embedding model into bit codes (random hyperplanes: Hamming distance ~ angle).
  With a weak stand-in embedding, hybrid + margin training scored 71.1% on the paraphrase benchmark (plain lexical 63.3%, LogReg 71.1%) - within noise.
  The real-model run is **untested** and is the next experiment to run.
- **Semantics (negative result):** random indexing learned from a local 4.4M-word docstring corpus did not improve few-shot paraphrase
  accuracy (hybrid 62-66% vs lexical 63%, within noise); IDF-weighted bundling made it worse. See `results/semantic.md`.
- **Procedural storage:** removes a 5.2 MB item table on this task but costs ~8.5 us per token regeneration (Python).

### Hypothesis status

| | Status |
|---|---|
| H1 procedural compression | Partly supported: storage saved, but equal to the hashing trick; learned prototypes are still stored |
| H2 discrete phase encoding | Supported for retrieval; binary matched or beat phase at equal bits on every test |
| H3 event-driven efficiency | Supported on x86 in the low-activity regime; break-even is ~70-95% activity |
| H4 hybrid semantic performance | **Not met** against the pre-registered criterion (within 2 points of int8 baseline): gap 3.4 points at 80 shots |
| H5 compositional generalization | Supported for known words (89% with margin training, optimistic; LogReg 95%); fails for new synonyms (7-14%) |
| Phase D multi-step association (new) | Supported on a synthetic graph: multi-hop 99.7-100%, corruption-tolerant; storage cost 4-15x an explicit table |

## Limitations (read these)

- All accuracy numbers are from a **synthetic** benchmark generated by `make_synthetic()`; real-language results are untested.
- The knowledge-memory graph is **synthetic and random**: entities are opaque names, relations are functional, and no text-to-triple extraction is attempted.
- The text encoder is **lexical** (word + character-trigram hashing). There is no pretrained semantic encoder yet, which is
  the most likely route to closing the accuracy gap.
- Cascade noise experiments use **i.i.d. bit flips**; the adaptive margin rule needs validation on real encoder output.
- All speed numbers are from one x86 core. **ARM and energy are not measured.** Python latencies include interpreter and
  library overhead and do not predict optimized edge code.
- Recall figures in the C sweeps use 100 queries each (a "100%" has a lower bound of roughly 96-97%).
- No claims are made about general-purpose language generation or reasoning.

## Suggested next steps

0. Get real language data into the loop. Both semantic attempts failed for lack of it, so this is the main blocker for H4.
1. Run `make experiments-real` on Banking77 / CLINC150 and compare with the synthetic results.
2. Run `experiments/run_embedding.py` with a real sentence model (the hook is written; the run is not) and re-test H4.
3. Run `make bench` and `make sweep` on the real target device (ARM), with a power meter for energy per query.
4. Replace the hand-built phrasing lists in `run_qa.py` with real questions/statements, and swap the lexical reader for a pretrained encoder - the memory side already works.
