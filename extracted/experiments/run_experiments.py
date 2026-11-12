"""
run_experiments.py - runs the PS-HR validation suite and writes results.json + REPORT.md

Usage:
  python run_experiments.py                         # synthetic intent benchmark
  python run_experiments.py --data train.tsv test.tsv   # real data: 'label<TAB>text' per line
                                                    # (e.g. Banking77 / CLINC150 exported to TSV)
"""
import argparse, json, os, random, string, time, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
import scipy.sparse as sp

from pshr import (BinaryHDC, PhaseHDC, HDClassifier, SplitStore, ProceduralItemMemory,
                  text_features, seed_of, expand_seed, popcount_rows, pack_bits)

# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def pseudo_word(rng):
    syl = ["ka", "vro", "mi", "tel", "so", "ran", "du", "pei", "lun", "zo", "fa", "ni", "bar", "qui", "ex"]
    return "".join(rng.choice(syl) for _ in range(rng.randint(2, 3)))


def typo(w, rng):
    if len(w) < 4:
        return w
    i = rng.randrange(len(w))
    op = rng.choice("sdi")
    if op == "s":
        return w[:i] + rng.choice(string.ascii_lowercase) + w[i + 1:]
    if op == "d":
        return w[:i] + w[i + 1:]
    return w[:i] + rng.choice(string.ascii_lowercase) + w[i:]


def make_synthetic(n_intents=40, group=4, shots=80, n_test=30, seed=0, typo_rate=0.12):
    """Confusable-intent generator. Intents in a group share 2 of their 4 signature words;
    each utterance uses 3 of 4 signature words, filler words, and sometimes a distractor
    signature word from another intent. Typos injected in train and test."""
    rng = random.Random(seed)
    filler = [pseudo_word(rng) for _ in range(150)]
    frames = ["please {} {} {}", "i want to {} {} about {}", "can you {} the {} for {}",
              "how do i {} my {} {}", "{} {} {} now", "need help to {} {} with {}"]
    sig = {}
    for g in range(n_intents // group):
        shared = [pseudo_word(rng) for _ in range(2)]
        for k in range(group):
            sig[g * group + k] = shared + [pseudo_word(rng) for _ in range(2)]

    def utt(c):
        words = rng.sample(sig[c], 3)
        if rng.random() < 0.25:
            words[rng.randrange(3)] = rng.choice(sig[rng.randrange(n_intents)])
        extra = [rng.choice(filler) for _ in range(rng.randint(0, 3))]
        s = rng.choice(frames).format(*words)
        toks = (s.split() + extra)
        rng.shuffle(toks)
        return " ".join(typo(t, rng) if rng.random() < typo_rate else t for t in toks)

    tr = [(c, utt(c)) for c in range(n_intents) for _ in range(shots)]
    te = [(c, utt(c)) for c in range(n_intents) for _ in range(n_test)]
    rng.shuffle(tr); rng.shuffle(te)
    return tr, te


def load_tsv(path):
    rows = [l.rstrip("\n").split("\t", 1) for l in open(path, encoding="utf8") if "\t" in l]
    return rows


def subsample(train, shots, seed):
    rng = random.Random(seed)
    by = {}
    for c, t in train:
        by.setdefault(c, []).append(t)
    out = []
    for c, ts in by.items():
        rng.shuffle(ts)
        out += [(c, t) for t in ts[:shots]]
    return out


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------
class SparseLR:
    """Hashed word(1-2)+char(2-4) TF-IDF -> logistic regression. Optional int8 weights."""

    def __init__(self, nf=2 ** 14, int8=False):
        self.nf, self.int8 = nf, int8
        self.hw = HashingVectorizer(n_features=nf, analyzer="word", ngram_range=(1, 2), alternate_sign=False, norm=None)
        self.hc = HashingVectorizer(n_features=nf, analyzer="char_wb", ngram_range=(2, 4), alternate_sign=False, norm=None)
        self.tf = TfidfTransformer(sublinear_tf=True)
        self.name = f"LogReg-hash{'-int8' if int8 else '-f32'}(2^{int(np.log2(nf))}x2)"

    def _x(self, texts):
        return sp.hstack([self.hw.transform(texts), self.hc.transform(texts)]).tocsr()

    def fit(self, texts, y):
        X = self.tf.fit_transform(self._x(texts))
        self.lr = LogisticRegression(max_iter=300, C=20).fit(X, y)
        W = self.lr.coef_.astype(np.float32)
        if self.int8:
            self.scale = np.abs(W).max(1, keepdims=True) / 127 + 1e-12
            self.Wq = np.rint(W / self.scale).astype(np.int8)
        self.W, self.b = W, self.lr.intercept_.astype(np.float32)
        return self

    def predict(self, texts):
        X = self.tf.transform(self._x(texts))
        if self.int8:
            s = (X @ (self.Wq.astype(np.float32) * self.scale).T) + self.b
        else:
            s = X @ self.W.T + self.b
        return self.lr.classes_[np.asarray(s).argmax(1)]

    def model_bytes(self):
        return self.Wq.nbytes + self.scale.nbytes if self.int8 else self.W.nbytes


class TfidfCentroid:
    name = "TF-IDF nearest-centroid(hash 2^14x2)"

    def __init__(self, nf=2 ** 14):
        self.lr = SparseLR(nf)

    def fit(self, texts, y):
        X = self.lr.tf.fit_transform(self.lr._x(texts))
        y = np.asarray(y)
        self.cls = np.unique(y)
        C = np.vstack([np.asarray(X[y == c].mean(0)) for c in self.cls]).astype(np.float32)
        self.C = normalize(C)
        return self

    def predict(self, texts):
        X = normalize(self.lr.tf.transform(self.lr._x(texts)))
        return self.cls[np.asarray(X @ self.C.T).argmax(1)]

    def model_bytes(self):
        return self.C.nbytes


def acc(pred, y):
    return float((np.asarray(pred) == np.asarray(y)).mean())


# --------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------
def exp_learning_curve(train, test, C, seeds, shots_list, D=8192, extra_encoders=None):
    res = {}
    ytest = np.array([c for c, _ in test]); Xte_txt = [t for _, t in test]
    for shots in shots_list:
        row = {}
        for s in seeds:
            sub = subsample(train, shots, s)
            y = np.array([c for c, _ in sub]); txt = [t for _, t in sub]
            models = {}
            # HDC variants
            cfgs = [("BinaryHDC-8192b-noretrain", BinaryHDC(D, salt=s), 0, 0.0),
                    ("BinaryHDC-8192b-retrain", BinaryHDC(D, salt=s), 15, 0.0),
                    ("BinaryHDC-8192b-margin0.12", BinaryHDC(D, salt=s), 60, 0.12),
                    ("PhaseHDC-Q16-8192b-retrain", PhaseHDC(D, 16, salt=s), 15, 0.0)]
            for label, fac in (extra_encoders or []):      # each extra encoder is run both plain and margin-trained
                cfgs.append((label + ", plain", fac(), 15, 0.0))
                cfgs.append((label + ", margin0.12", fac(), 60, 0.12))
            for label, enc, ep, mg in cfgs:
                if hasattr(enc, "fit_center"): enc.fit_center(txt)
                elif hasattr(enc, "emb"): enc.emb.fit_center(txt)
                Xtr = enc.encode_many(txt); Xte = enc.encode_many(Xte_txt)
                clf = HDClassifier(enc, ep, mg).fit(Xtr, y, C)
                row.setdefault(label, []).append(acc(clf.predict(Xte), ytest))
            for m in [SparseLR(2 ** 14), SparseLR(2 ** 14, int8=True), SparseLR(2 ** 10), TfidfCentroid()]:
                m.fit(txt, y)
                row.setdefault(m.name, []).append(acc(m.predict(Xte_txt), ytest))
        res[shots] = {k: (float(np.mean(v)), float(np.std(v))) for k, v in row.items()}
    return res


def exp_model_size_latency(train, test, C, shots=80, D=8192, n_q=300):
    sub = subsample(train, shots, 0)
    y = np.array([c for c, _ in sub]); txt = [t for _, t in sub]
    qs = [t for _, t in test][:n_q]
    out = {}

    def bench(fn):
        fn(qs[0])
        t0 = time.perf_counter()
        for q in qs:
            fn(q)
        return (time.perf_counter() - t0) / len(qs) * 1e6

    for label, enc in [("BinaryHDC", BinaryHDC(D)), ("PhaseHDC-Q16", PhaseHDC(D, 16))]:
        clf = HDClassifier(enc, 15).fit(enc.encode_many(txt), y, C)
        if label == "BinaryHDC":
            fn = lambda q: popcount_rows(clf.protos ^ enc.encode(q)).argmin()
        else:
            fn = lambda q: enc.scores_lut(clf.protos, enc.encode(q)).argmax()
        # encode-only latency
        t_enc = bench(lambda q: enc.encode(q))
        out[label] = dict(model_bytes=enc.model_bytes(C), latency_us=bench(fn), encode_us=t_enc)
    for m in [SparseLR(2 ** 14), SparseLR(2 ** 14, int8=True), SparseLR(2 ** 10)]:
        m.fit(txt, y)
        out[m.name] = dict(model_bytes=int(m.model_bytes()), latency_us=bench(lambda q: m.predict([q])))
    return out


def exp_procedural_storage(train, D=8192):
    """H1: stored item-memory table vs regenerated from seed."""
    enc = BinaryHDC(D)
    feats = set()
    for _, t in train:
        toks, bg = text_features(t)
        feats.update(toks)
        feats.update(a for a, _ in bg); feats.update(b for _, b in bg)
    vocab = sorted(feats)
    table = {t: expand_seed(seed_of(t), D // 64) for t in vocab}
    # verify determinism
    assert all(np.array_equal(table[t], enc.im.regenerate(t)) for t in vocab[:200])
    sample = vocab[:2000]
    t0 = time.perf_counter()
    for t in sample:
        enc.im.regenerate(t)
    regen_us = (time.perf_counter() - t0) / len(sample) * 1e6
    t0 = time.perf_counter()
    for t in sample:
        table[t]
    look_us = (time.perf_counter() - t0) / len(sample) * 1e6
    return dict(n_features=len(vocab), table_bytes=len(vocab) * D // 8, seed_bytes=0,
                regen_us=regen_us, lookup_us=look_us)


def exp_noise(train, test, C, D=8192, shots=80, ps=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5)):
    sub = subsample(train, shots, 0)
    y = np.array([c for c, _ in sub]); txt = [t for _, t in sub]
    ytest = np.array([c for c, _ in test]); te = [t for _, t in test]
    rng = np.random.default_rng(0)
    out = {}
    for label, enc in [("BinaryHDC", BinaryHDC(D)), ("PhaseHDC-Q16", PhaseHDC(D, 16))]:
        clf = HDClassifier(enc, 15).fit(enc.encode_many(txt), y, C)
        Xte = enc.encode_many(te)
        out[label] = {p: acc(clf.predict(enc.corrupt(Xte, p, rng)), ytest) for p in ps}
    return out


def exp_capacity(bits_list=(1024, 2048, 4096, 8192), ks=(8, 16, 32, 64, 128, 256), cb=512, trials=3):
    """Holographic key-value memory: M = bundle(bind(K_i, V_i)); query unbind(M, K_j) -> cleanup vs value codebook."""
    out = {"binary": {}, "phase": {}}
    rng = np.random.default_rng(1)
    for bits in bits_list:
        for kind in ("binary", "phase"):
            row = {}
            for k in ks:
                accs = []
                for _ in range(trials):
                    if kind == "binary":
                        D = bits
                        K = rng.integers(0, 2, (k, D), dtype=np.uint8)
                        V = rng.integers(0, 2, (cb, D), dtype=np.uint8)
                        bound = K ^ V[:k]
                        ones = bound.sum(0)
                        M = np.where(2 * ones == k, rng.integers(0, 2, D), (2 * ones > k)).astype(np.uint8)
                        U = M[None] ^ K
                        sims = -(U[:, None, :] != V[None]).sum(-1)
                    else:
                        D = bits // 4
                        K = rng.integers(0, 16, (k, D)); V = rng.integers(0, 16, (cb, D))
                        z = np.exp(2j * np.pi * ((K + V[:k]) % 16) / 16).sum(0)
                        M = np.rint(np.angle(z) / (2 * np.pi / 16)).astype(int) % 16
                        U = (M[None] - K) % 16
                        zu = np.exp(2j * np.pi * U / 16); zv = np.exp(2j * np.pi * V / 16)
                        sims = (zu @ zv.conj().T).real
                    accs.append(float((sims.argmax(1) == np.arange(k)).mean()))
                row[k] = float(np.mean(accs))
            out[kind][bits] = row
    return out


def exp_cascade_scale(Ns=(1_000, 10_000, 100_000), W=128, w1=16, noise=0.2, margins=(0, 40, 80, 160), nq=15, seed=2):
    rng = np.random.default_rng(seed)
    out = {}
    for N in Ns:
        P = rng.integers(0, 2 ** 63, (N, W), dtype=np.uint64) ^ (rng.integers(0, 2 ** 63, (N, W), dtype=np.uint64) << np.uint64(1))
        store = SplitStore(P, w1)
        idx = rng.integers(0, N, nq)
        flips = pack_bits((rng.random((nq, W * 64)) < noise).astype(np.uint8))
        Q = P[idx] ^ flips
        # dense
        t0 = time.perf_counter()
        d = [store.dense(q)[0] for q in Q]
        t_dense = (time.perf_counter() - t0) / nq * 1e3
        row = {"dense": dict(ms=t_dense, recall=float(np.mean(np.array(d) == idx)), ops_frac=1.0, active=1.0)}
        for m in margins:
            t0 = time.perf_counter()
            r = [store.cascade(q, m) for q in Q]
            t = (time.perf_counter() - t0) / nq * 1e3
            row[f"cascade_m{m}"] = dict(ms=t, recall=float(np.mean([a[0] for a in r] == idx)),
                                        ops_frac=float(np.mean([a[1] for a in r]) / (N * W)),
                                        active=float(np.mean([a[2] for a in r])))
        out[N] = row
        del P, store
    return out


def exp_cascade_classes(train, test, C, D=8192, shots=80):
    """Class-level cascade on the intent task (small C, so expect little gain)."""
    sub = subsample(train, shots, 0)
    y = np.array([c for c, _ in sub]); txt = [t for _, t in sub]
    ytest = np.array([c for c, _ in test]); te = [t for _, t in test]
    enc = BinaryHDC(D)
    clf = HDClassifier(enc, 15).fit(enc.encode_many(txt), y, C)
    Xte = enc.encode_many(te)
    store = SplitStore(clf.protos, 16)
    out = {}
    base = acc([store.dense(q)[0] for q in Xte], ytest)
    out["dense"] = dict(acc=base, ops_frac=1.0)
    for m in (0, 20, 40, 80, 160):
        r = [store.cascade(q, m) for q in Xte]
        out[f"m{m}"] = dict(acc=acc([a[0] for a in r], ytest), ops_frac=float(np.mean([a[1] for a in r]) / (C * store.W)),
                            active=float(np.mean([a[2] for a in r])))
    return out


# --------------------------------------------------------------------------
def md_report(R, data_desc):
    L = []
    A = L.append
    A("# PS-HR Prototype: Measured Results\n")
    A(f"Data: {data_desc}. Hardware: {R['env']}.\n")
    A("All numbers are produced by `experiments/run_experiments.py`; rerun `make experiments` to reproduce. Single-threaded NumPy; latencies include Python overhead and are NOT representative of optimized C on an edge CPU.\n")
    A("## 1. Accuracy vs training shots per class (mean ± std over seeds)\n")
    names = list(next(iter(R["learning"].values())).keys())
    A("| Model | " + " | ".join(f"{s} shots" for s in R["learning"]) + " |")
    A("|---|" + "---|" * len(R["learning"]))
    for n in names:
        A(f"| {n} | " + " | ".join(f"{R['learning'][s][n][0]*100:.1f} ± {R['learning'][s][n][1]*100:.1f}" for s in R["learning"]) + " |")
    A("\n## 2. Model size and per-query latency (80 shots)\n")
    A("| Model | Model bytes | Latency (µs/query, incl. encoding) |")
    A("|---|---|---|")
    for n, v in R["size"].items():
        A(f"| {n} | {v['model_bytes']:,} | {v['latency_us']:.0f} |")
    A("\n## 3. H1: procedural item memory\n")
    p = R["procedural"]
    A(f"- Distinct features seen: {p['n_features']:,}")
    A(f"- Stored item-memory table would cost: {p['table_bytes']:,} bytes; procedural: {p['seed_bytes']} bytes (seed = hash of the token string)")
    A(f"- Regeneration latency: {p['regen_us']:.1f} µs/token vs dict lookup {p['lookup_us']:.2f} µs/token")
    A("- Caveat: only the random item vectors are procedural. The class prototypes are learned and are stored.\n")
    A("## 4. Noise robustness (accuracy when a fraction of query components is randomized)\n")
    ps = list(next(iter(R["noise"].values())).keys())
    A("| Model | " + " | ".join(f"p={p}" for p in ps) + " |")
    A("|---|" + "---|" * len(ps))
    for n, v in R["noise"].items():
        A(f"| {n} | " + " | ".join(f"{v[p]*100:.1f}" for p in ps) + " |")
    A("\n## 5. Holographic key-value capacity (retrieval accuracy at equal bit budget)\n")
    for kind in ("binary", "phase"):
        A(f"**{kind}**\n")
        ks = list(next(iter(R["capacity"][kind].values())).keys())
        A("| bits | " + " | ".join(f"k={k}" for k in ks) + " |")
        A("|---|" + "---|" * len(ks))
        for bits, row in R["capacity"][kind].items():
            A(f"| {bits} | " + " | ".join(f"{row[k]*100:.0f}" for k in ks) + " |")
        A("")
    A("## 6. H3: event-driven cascade, large associative store (20% query noise, 8192-bit vectors)\n")
    A("| N items | mode | ms/query | recall | ops fraction | nodes activated |")
    A("|---|---|---|---|---|---|")
    for N, rows in R["cascade_scale"].items():
        for mode, v in rows.items():
            A(f"| {int(N):,} | {mode} | {v['ms']:.2f} | {v['recall']*100:.0f}% | {v['ops_frac']:.3f} | {v['active']*100:.1f}% |")
    A("\n## 7. Cascade on the class prototypes of the intent task\n")
    A("| mode | accuracy | ops fraction |")
    A("|---|---|---|")
    for m, v in R["cascade_classes"].items():
        A(f"| {m} | {v['acc']*100:.1f}% | {v['ops_frac']:.2f} |")
    return "\n".join(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs=2, metavar=("TRAIN", "TEST"))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--embed-model", default=None, help="add embedding/hybrid encoders to the learning curve: 'ri' (stand-in) or a sentence-transformers name (untested)")
    ap.add_argument("--out", default="../results", help="output directory for results.json and REPORT.md")
    a = ap.parse_args()
    import platform
    env = f"{platform.processor() or platform.machine()}, 1 thread, Python {platform.python_version()}, NumPy {np.__version__}"
    if a.data:
        tr, te = load_tsv(a.data[0]), load_tsv(a.data[1])
        labels = {l: i for i, l in enumerate(sorted({l for l, _ in tr}))}
        tr = [(labels[l], t) for l, t in tr]; te = [(labels[l], t) for l, t in te if l in labels]
        desc = f"real data from {a.data}"
        shots_list = (5, 20, 80)
    else:
        tr, te = make_synthetic()
        desc = "SYNTHETIC 40-intent confusable benchmark (pseudo-words, 12% typo rate, 25% distractor words); not a real-language benchmark"
        shots_list = (5, 20, 80)
    C = len({c for c, _ in tr})
    R = {"env": env}
    t = time.time()
    extras = None
    if a.embed_model:
        from embed_hdc import EmbeddingHDC, HybridEncoder, load_embedder
        embed_fn, edim = load_embedder(a.embed_model)
        extras = [(f"EmbeddingHDC-2048b [{a.embed_model}]", lambda: EmbeddingHDC(embed_fn, edim, D=2048)),
                  (f"Hybrid lexical8192+2x emb2048 [{a.embed_model}]", lambda: HybridEncoder(BinaryHDC(8192), EmbeddingHDC(embed_fn, edim, D=2048), 2))]
    print("learning curve...", flush=True); R["learning"] = exp_learning_curve(tr, te, C, range(a.seeds), shots_list, extra_encoders=extras)
    print("size/latency...", flush=True);    R["size"] = exp_model_size_latency(tr, te, C)
    print("procedural...", flush=True);      R["procedural"] = exp_procedural_storage(tr)
    print("noise...", flush=True);           R["noise"] = exp_noise(tr, te, C)
    print("capacity...", flush=True);        R["capacity"] = exp_capacity()
    print("cascade scale...", flush=True);   R["cascade_scale"] = exp_cascade_scale()
    print("cascade classes...", flush=True); R["cascade_classes"] = exp_cascade_classes(tr, te, C)
    R["runtime_s"] = time.time() - t
    os.makedirs(a.out, exist_ok=True)
    json.dump(R, open(os.path.join(a.out, "results.json"), "w"), indent=1, default=str)
    open(os.path.join(a.out, "REPORT.md"), "w").write(md_report(R, desc))
    print(open(os.path.join(a.out, "REPORT.md")).read())
