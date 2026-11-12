"""
run_semantic.py - negative-result experiments on adding semantics to the lexical HDC encoder.

 (1) Random indexing (PPMI co-occurrence -> seeded random projection -> binarized word hypervectors)
     learned from a LOCAL corpus (python/numpy/scipy/sklearn docstrings; see scripts/build_corpus.py)
 (2) IDF-weighted bundling in the lexical encoder (weights from the same corpus)
evaluated on a small hand-written few-shot paraphrase benchmark (experiments/intents_sysadmin.py).

Prerequisite:  python scripts/build_corpus.py   (writes corpus.txt in the current directory)
Usage (from experiments/):  python run_semantic.py
The corpus depends on what is installed locally, so exact numbers will differ between machines.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import re, numpy as np, collections
import scipy.sparse as sp
from pshr import BinaryHDC, HDClassifier, pack_bits
import intents_sysadmin as D
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
import numpy as np, scipy.sparse as sp, time, collections, re, collections
def build(corpus="corpus.txt", V=10000, min_count=20, window=4, d=2048, seed=7, alpha=0.75):
    t0 = time.time()
    cnt = collections.Counter()
    lines = [l.split() for l in open(corpus)]
    for l in lines: cnt.update(l)
    vocab = [w for w, c in cnt.most_common(V) if c >= min_count and len(w) > 1]
    idx = {w: i for i, w in enumerate(vocab)}; Vn = len(vocab)
    rows, cols, vals = [], [], []
    for l in lines:
        ids = np.array([idx.get(w, -1) for w in l], dtype=np.int32)
        for k in range(1, window + 1):
            a, b = ids[:-k], ids[k:]
            m = (a >= 0) & (b >= 0)
            if m.any():
                w = 1.0 / k
                rows += [a[m], b[m]]; cols += [b[m], a[m]]; vals += [np.full(m.sum(), w, np.float32)] * 2
    X = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(Vn, Vn)).tocsr()
    X.sum_duplicates()
    # PPMI with context-distribution smoothing
    rs = np.asarray(X.sum(1)).ravel(); cs = np.asarray(X.sum(0)).ravel() ** alpha
    tot = rs.sum(); cs = cs / cs.sum() * tot
    X = X.tocoo()
    pmi = np.log((X.data * tot) / (rs[X.row] * cs[X.col]) + 1e-12)
    keep = pmi > 0
    P = sp.coo_matrix((pmi[keep].astype(np.float32), (X.row[keep], X.col[keep])), shape=(Vn, Vn)).tocsr()
    R = np.random.default_rng(seed).standard_normal((Vn, d)).astype(np.float32) / np.sqrt(d)   # procedural: regenerated from `seed`
    E = P @ R
    E -= E.mean(0, keepdims=True)
    E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    # frequency-based IDF-ish weight
    freq = np.array([cnt[w] for w in vocab], dtype=np.float64)
    weight = np.log((freq.sum() / freq)).astype(np.float32)
    print(f"built V={Vn} d={d} nnz={P.nnz} in {time.time()-t0:.0f}s", flush=True)
    return vocab, idx, E, weight

def ensure_ri():
    if not os.path.exists("ri.npz"):
        vocab, idx, E, w = build()
        np.savez("ri.npz", E=E, weight=w, vocab=np.array(vocab))

ensure_ri()
print("=== (1) random-indexing semantic encoder ===")
z = np.load("ri.npz"); E = z["E"]; wt = z["weight"]; vocab = list(z["vocab"]); idx = {w: i for i, w in enumerate(vocab)}
Bpm = np.where(E > 0, 1, -1).astype(np.int8)       # binarized word codes (d bits each)
d = E.shape[1]
W_RE = re.compile(r"[a-z]+")

def known(text): return [idx[w] for w in W_RE.findall(text.lower()) if w in idx]

class SemanticBits:
    """Text -> d-bit hypervector: weighted bundle of binarized random-indexing word codes."""
    def __init__(self): self.D, self.W = d, d // 64
    def encode(self, text):
        ids = known(text)
        if not ids: return np.zeros(self.W, dtype=np.uint64)
        s = (Bpm[ids].astype(np.float32) * wt[ids, None]).sum(0)
        return pack_bits((s > 0).astype(np.uint8)).reshape(-1)

class Concat(BinaryHDC):
    """lexical (8192b) + r copies of the semantic (2048b) segment, compared by one Hamming distance."""
    def __init__(self, r=1, lex=True):
        self.lex = BinaryHDC(8192) if lex else None; self.sem = SemanticBits(); self.r = r
        self.D = (8192 if lex else 0) + r * d; self.W = self.D // 64
    def encode(self, text):
        parts = ([self.lex.encode(text)] if self.lex else []) + [self.sem.encode(text)] * self.r
        return np.concatenate(parts)
    def encode_many(self, texts): return np.stack([self.encode(t) for t in texts])

def run_hdc(enc, epochs=0):
    ytr = np.array([c for c, _ in D.train]); yte = np.array([c for c, _ in D.test])
    Xtr = enc.encode_many([t for _, t in D.train]); Xte = enc.encode_many([t for _, t in D.test])
    clf = HDClassifier(enc, epochs).fit(Xtr, ytr, len(D.labels))
    return float((clf.predict(Xte) == yte).mean())

ytr = np.array([c for c, _ in D.train]); yte = np.array([c for c, _ in D.test])
res = {}
res["Lexical HDC (8192b)"] = run_hdc(BinaryHDC(8192))
res["Semantic HDC only (2048b, random indexing)"] = run_hdc(Concat(1, lex=False))
for r in (1, 2, 4, 8):
    res[f"Hybrid HDC (8192b lexical + {r}x2048b semantic)"] = run_hdc(Concat(r))
res["Hybrid, +retrain 15 ep (r=4)"] = run_hdc(Concat(4), 15)

# float baselines
def mean_emb(t):
    ids = known(t)
    return (E[ids] * wt[ids, None]).sum(0) / (np.linalg.norm((E[ids] * wt[ids, None]).sum(0)) + 1e-9) if ids else np.zeros(d, np.float32)
Xtr_e = np.stack([mean_emb(t) for _, t in D.train]); Xte_e = np.stack([mean_emb(t) for _, t in D.test])
cent = np.stack([Xtr_e[ytr == c].mean(0) for c in range(len(D.labels))])
res["Float RI centroid (cosine, same embeddings)"] = float(((Xte_e @ cent.T).argmax(1) == yte).mean())
res["Float RI + LogReg"] = float((LogisticRegression(max_iter=500, C=20).fit(Xtr_e, ytr).predict(Xte_e) == yte).mean())
hv = HashingVectorizer(n_features=2**14, analyzer="word", ngram_range=(1, 2), alternate_sign=False, norm=None)
hc = HashingVectorizer(n_features=2**14, analyzer="char_wb", ngram_range=(2, 4), alternate_sign=False, norm=None)
tf = TfidfTransformer(sublinear_tf=True)
fx = lambda ts: sp.hstack([hv.transform(ts), hc.transform(ts)]).tocsr()
Xl = tf.fit_transform(fx([t for _, t in D.train]))
res["Lexical TF-IDF + LogReg"] = float((LogisticRegression(max_iter=500, C=20).fit(Xl, ytr).predict(tf.transform(fx([t for _, t in D.test]))) == yte).mean())
res["Chance"] = 1 / len(D.labels)

cov = np.mean([len(known(t)) / max(1, len(W_RE.findall(t.lower()))) for _, t in D.test])
print(f"vocab coverage of test words: {cov*100:.0f}%   (n_test={len(D.test)}, 1 item = {100/len(D.test):.2f} pts, binomial SE ~ 3-4 pts)\n")
for k, v in res.items(): print(f"{v*100:5.1f}%  {k}")

print("\n=== (2) IDF-weighted lexical bundling (exploratory; swept against the same 128 test items) ===")
cnt = collections.Counter()
for l in open("corpus.txt"): cnt.update(l.split())
TOT = sum(cnt.values())
def idf(word): return np.log(TOT / (cnt.get(word, 0) + 1))

class WeightedBinary(BinaryHDC):
    """Binary HDC with IDF-weighted bundling. Token weight = idf(word)**p; char trigrams inherit their word's weight."""
    def __init__(self, D=8192, p=1.0, trig=1.0, bigram=1.0, floor=0.0):
        super().__init__(D); self.p, self.trig, self.bigram, self.floor = p, trig, bigram, floor
    def encode(self, text):
        words = re.findall(r"[a-z0-9']+", text.lower())
        rows, ws = [], []
        for w in words:
            wt = max(idf(w) - self.floor, 0.01) ** self.p
            rows.append(self.im.get("w:" + w)); ws.append(wt)
            pw = f"#{w}#"
            for i in range(len(pw) - 2):
                rows.append(self.im.get("c:" + pw[i:i+3])); ws.append(wt * self.trig)
        for a, b in zip(words, words[1:]):
            rows.append(np.roll(self.im.get("w:" + a), 1) ^ self.im.get("w:" + b)); ws.append(self.bigram * (max(idf(a)-self.floor,.01)*max(idf(b)-self.floor,.01)) ** (self.p / 2))
        mat = np.ascontiguousarray(np.stack(rows)); w = np.array(ws, dtype=np.float32)
        bits = np.unpackbits(mat.view(np.uint8), axis=1).astype(np.float32)
        s = (bits * w[:, None]).sum(0) - w.sum() / 2
        tie = np.unpackbits(self.tie.view(np.uint8))
        out = np.where(np.abs(s) < 1e-6, tie, (s > 0).astype(np.uint8))
        return pack_bits(out).reshape(-1)
    def encode_many(self, texts): return np.stack([self.encode(t) for t in texts])

def acc_idf(enc, ep=0):
    Xtr = enc.encode_many([t for _, t in D.train]); Xte = enc.encode_many([t for _, t in D.test])
    return (HDClassifier(enc, ep).fit(Xtr, ytr, len(D.labels)).predict(Xte) == yte).mean() * 100
print("baseline unweighted lexical:", round(acc_idf(BinaryHDC(8192)), 1))
for p in (1, 2, 3):
    for trig in (0.0, 0.3, 1.0):
        print(f"idf^{p}  trigram weight {trig}: {acc_idf(WeightedBinary(8192, p, trig)):.1f}%")
for floor in (0, 3, 5):
    print(f"idf^2 trig 0.3 floor {floor}: {acc_idf(WeightedBinary(8192, 2, 0.3, 1.0, floor)):.1f}%")
