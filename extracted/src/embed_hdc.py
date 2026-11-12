"""
embed_hdc.py - plug a (pretrained) sentence-embedding model into the HDC machinery.

  text --embed_fn--> float vector --(centre, random hyperplanes)--> D bits --> same prototypes / memory as before

Random-hyperplane binarization (a.k.a. sign random projection): the fraction of differing bits between two codes
is angle(x, y) / pi, so Hamming distance on the codes approximates cosine similarity of the embeddings.
The hyperplanes are regenerated from a seed (procedural), so only the seed needs storing.

STATUS: the binarization and hybrid logic are unit-tested with stand-in embeddings. The sentence-transformers adapter
below could NOT be run in the sandbox where this was written (no network, no model); treat it as untested.
"""
import numpy as np
from pshr import BinaryHDC, pack_bits


class EmbeddingHDC(BinaryHDC):
    """Embedding model -> binary hypervector. Subclasses BinaryHDC so classifiers/memory work unchanged."""
    name = "EmbeddingHDC"

    def __init__(self, embed_fn, dim, D=2048, seed=0):
        assert D % 64 == 0
        self.embed_fn, self.dim, self.D, self.W = embed_fn, dim, D, D // 64
        self.R = np.random.default_rng(seed).standard_normal((dim, D)).astype(np.float32)   # procedural: from `seed`
        self.mean = np.zeros(dim, dtype=np.float32)

    def fit_center(self, texts):
        """Embedding models are anisotropic (all vectors share a large common component); subtract the mean."""
        self.mean = np.asarray(self.embed_fn(list(texts)), dtype=np.float32).mean(0)
        return self

    def encode_many(self, texts):
        X = np.asarray(self.embed_fn(list(texts)), dtype=np.float32) - self.mean
        return np.ascontiguousarray(pack_bits((X @ self.R > 0).astype(np.uint8))).reshape(len(texts), self.W)

    def encode(self, text):
        return self.encode_many([text])[0]


class HybridEncoder(BinaryHDC):
    """lexical HDC (typo/word matching) + r copies of an embedding code (meaning), compared by one Hamming distance."""
    name = "HybridEncoder"

    def __init__(self, lexical, embedding, r=2):
        self.lex, self.emb, self.r = lexical, embedding, r
        self.D = lexical.D + r * embedding.D
        self.W = self.D // 64

    def encode_many(self, texts):
        L = self.lex.encode_many(texts)
        E = self.emb.encode_many(texts)
        return np.concatenate([L] + [E] * self.r, axis=1)

    def encode(self, text):
        return self.encode_many([text])[0]


def sentence_transformers_embed(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """UNTESTED adapter (needs `pip install sentence-transformers` and network on first run).
    Returns (embed_fn, dim)."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    dim = model.get_sentence_embedding_dimension()
    return (lambda texts: model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)), dim


def load_embedder(name):
    """name == 'ri': local random-indexing stand-in (needs ri.npz in the working dir, see experiments/run_semantic.py);
    otherwise a sentence-transformers model name (UNTESTED here). Returns (embed_fn, dim); results are cached per text."""
    import re
    if name == "ri":
        z = np.load("ri.npz"); E, wt = z["E"], z["weight"]; idx = {w: i for i, w in enumerate(z["vocab"])}
        def raw(texts):
            out = np.zeros((len(texts), E.shape[1]), np.float32)
            for k, t in enumerate(texts):
                ids = [idx[w] for w in re.findall(r"[a-z]+", t.lower()) if w in idx]
                if ids:
                    out[k] = (E[ids] * wt[ids, None]).sum(0)
            return out
        fn, dim = raw, E.shape[1]
    else:
        fn, dim = sentence_transformers_embed(name)
    cache = {}
    def cached(texts):
        texts = list(texts); miss = [t for t in dict.fromkeys(texts) if t not in cache]
        if miss:
            for t, v in zip(miss, fn(miss)):
                cache[t] = v
        return np.stack([cache[t] for t in texts])
    return cached, dim
