"""
run_embedding.py - evaluate lexical vs embedding vs hybrid encoders on (a) the 16-intent paraphrase benchmark and
(b) the end-to-end text QA task with unseen phrasings.

  python run_embedding.py --model ri                                   # stand-in: local random-indexing vectors (needs ri.npz, see run_semantic.py)
  python run_embedding.py --model sentence-transformers/all-MiniLM-L6-v2   # REAL model: UNTESTED here (needs network + sentence-transformers)

With --model ri this only verifies the plumbing (those embeddings are weak). The real question - whether a pretrained
sentence model closes the paraphrase gap - can only be answered on a machine that can load one.
"""
import argparse, os, sys, re
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pshr import BinaryHDC, HDClassifier
from embed_hdc import EmbeddingHDC, HybridEncoder, sentence_transformers_embed
import intents_sysadmin as D
import run_qa

ap = argparse.ArgumentParser(); ap.add_argument("--model", default="ri"); ap.add_argument("--margin", type=float, default=0.12)
ap.add_argument("--worlds", type=int, default=5)
a = ap.parse_args()

if a.model == "ri":
    z = np.load("ri.npz"); E, wt = z["E"], z["weight"]; idx = {w: i for i, w in enumerate(z["vocab"])}
    def embed_fn(texts):
        out = np.zeros((len(texts), E.shape[1]), np.float32)
        for k, t in enumerate(texts):
            ids = [idx[w] for w in re.findall(r"[a-z]+", t.lower()) if w in idx]
            if ids: out[k] = (E[ids] * wt[ids, None]).sum(0)
        return out
    dim = E.shape[1]
else:
    embed_fn, dim = sentence_transformers_embed(a.model)

def make(kind, r=2):
    lex = BinaryHDC(8192)
    if kind == "lexical": return lex
    emb = EmbeddingHDC(embed_fn, dim, D=2048)
    return emb if kind == "embedding" else HybridEncoder(lex, emb, r)

ytr = np.array([c for c, _ in D.train]); yte = np.array([c for c, _ in D.test])
trt = [t for _, t in D.train]; tet = [t for _, t in D.test]
print(f"=== (a) 16-intent paraphrase benchmark (model: {a.model}; 1 item = 0.78 pts, SE ~3-4 pts) ===")
for kind, r in (("lexical", 0), ("embedding", 0), ("hybrid", 1), ("hybrid", 2), ("hybrid", 4)):
    for margin in (0.0, a.margin):
        enc = make(kind, r)
        if hasattr(enc, "fit_center"): enc.fit_center(trt)
        elif hasattr(enc, "emb"): enc.emb.fit_center(trt)
        clf = HDClassifier(enc, 60 if margin else 15, margin).fit(enc.encode_many(trt), ytr, len(D.labels))
        print(f"  {kind:9s} r={r}  margin={margin:4.2f}: {(clf.predict(enc.encode_many(tet)) == yte).mean() * 100:5.1f}%")

print(f"\n=== (b) text QA, unseen phrasings (fixed test phrasing, {a.worlds} worlds) ===")
for kind, r in (("lexical", 0), ("hybrid", 2)):
    rs = [run_qa.run(s, False, 0, False, (lambda: make(kind, r)) if kind != "lexical" else None) for s in range(a.worlds)]
    f = lambda k: np.mean([x[0][k] for x in rs]) * 100
    print(f"  {kind:8s}: reading {f('rel_acc'):.1f}%  question-type {f('q_type'):.1f}%  end-to-end 1-step {f('e2e1'):.1f}%  2-step {f('e2e2'):.1f}%")
