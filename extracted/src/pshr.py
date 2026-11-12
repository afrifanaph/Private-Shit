"""
pshr.py - Reference implementation of the PS-HR prototype (Phase A components).

Components
  1. Procedural item memory   : 64-bit seed -> hypervector, zero stored table (H1)
  2. Binary HDC               : packed uint64, XOR bind, majority bundle, popcount similarity (H2)
  3. Phase HDC (FHRR, Q=16)   : discrete phases, bind = add mod Q, bundle = phasor sum (H2)
  4. Event-driven cascade     : prefix-match "pulse" gates which nodes get the full comparison (H3)

Pure NumPy (>=2.0 for np.bitwise_count). Integer/bitwise operations everywhere on the
inference path except the phase-similarity LUT, which is a float32 table lookup.
"""
import hashlib
import re
import numpy as np

U64 = np.uint64
_GOLD = U64(0x9E3779B97F4A7C15)


# --------------------------------------------------------------------------
# 1. Procedural generation: seed -> vector (counter-based splitmix64)
# --------------------------------------------------------------------------
def _splitmix64(x):
    with np.errstate(over="ignore"):
        x = x + _GOLD
        x = (x ^ (x >> U64(30))) * U64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> U64(27))) * U64(0x94D049BB133111EB)
        return x ^ (x >> U64(31))


def seed_of(token, salt=0):
    h = hashlib.blake2b(f"{salt}|{token}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "little")


def expand_seed(seed, n_words):
    """Deterministically expand a 64-bit seed into n_words uint64 words."""
    with np.errstate(over="ignore"):
        ctr = U64(seed) + np.arange(n_words, dtype=U64) * _GOLD
    return _splitmix64(ctr)


class ProceduralItemMemory:
    """Token -> hypervector, regenerated from a hash-derived seed. Bounded cache."""

    def __init__(self, n_words, kind="binary", cache_size=100_000, salt=0):
        self.n_words, self.kind, self.cache_size, self.salt = n_words, kind, cache_size, salt
        self.cache = {}
        self.hits = self.misses = 0

    def get(self, token):
        v = self.cache.get(token)
        if v is not None:
            self.hits += 1
            return v
        self.misses += 1
        words = expand_seed(seed_of(token, self.salt), self.n_words)
        v = words if self.kind == "binary" else (words.view(np.uint8) & 15)
        if len(self.cache) >= self.cache_size:
            self.cache.clear()
        self.cache[token] = v
        return v

    def regenerate(self, token):
        """Uncached generation (for latency measurement)."""
        words = expand_seed(seed_of(token, self.salt), self.n_words)
        return words if self.kind == "binary" else (words.view(np.uint8) & 15)


# --------------------------------------------------------------------------
# Text -> feature tokens (shared by both encoders)
# --------------------------------------------------------------------------
_WORD = re.compile(r"[a-z0-9']+")


def text_features(text, use_bigrams=True, use_trigrams=True):
    """Returns (unigram tokens, list of (w1,w2) bigrams). Char trigrams give typo tolerance.
    use_bigrams / use_trigrams allow ablation (word-pair features can hurt when combinations are novel)."""
    words = _WORD.findall(text.lower())
    toks = []
    for w in words:
        toks.append("w:" + w)
        p = f"#{w}#"
        if use_trigrams:
            toks.extend("c:" + p[i:i + 3] for i in range(len(p) - 2))
    bigrams = [("w:" + a, "w:" + b) for a, b in zip(words, words[1:])] if use_bigrams else []
    return toks, bigrams


# --------------------------------------------------------------------------
# 2. Binary HDC
# --------------------------------------------------------------------------
def popcount_rows(x):
    return np.bitwise_count(x).sum(axis=-1, dtype=np.int32)


def unpack_bipolar(packed):
    """(n,W) uint64 -> (n,D) int8 in {-1,+1}."""
    p = np.ascontiguousarray(np.atleast_2d(packed))
    bits = np.unpackbits(p.view(np.uint8), axis=1)
    return (bits.astype(np.int8) * 2 - 1)


def pack_bits(bits01):
    """(D,) or (n,D) {0,1} -> packed uint64."""
    b = np.packbits(bits01.astype(np.uint8), axis=-1)
    return np.ascontiguousarray(b).view(np.uint64)


class BinaryHDC:
    name = "BinaryHDC"

    def __init__(self, D=8192, salt=0, use_bigrams=True, use_trigrams=True):
        assert D % 64 == 0
        self.D, self.W = D, D // 64
        self.use_bigrams, self.use_trigrams = use_bigrams, use_trigrams
        self.im = ProceduralItemMemory(self.W, "binary", salt=salt)
        self.tie = expand_seed(seed_of("tiebreak", salt), self.W)
        self.model_bits_per_vec = D

    def encode(self, text):
        toks, bigrams = text_features(text, self.use_bigrams, self.use_trigrams)
        rows = [self.im.get(t) for t in toks]
        rows += [np.roll(self.im.get(a), 1) ^ self.im.get(b) for a, b in bigrams]  # order-aware bind
        if not rows:
            return self.tie.copy()
        mat = np.ascontiguousarray(np.stack(rows))
        n = mat.shape[0]
        ones = np.unpackbits(mat.view(np.uint8), axis=1).sum(axis=0, dtype=np.int32)
        tie_bits = np.unpackbits(self.tie.view(np.uint8))
        out = np.where(2 * ones == n, tie_bits, (2 * ones > n).astype(np.uint8))
        return pack_bits(out).reshape(-1)

    def encode_many(self, texts):
        return np.stack([self.encode(t) for t in texts])

    def scores(self, protos, X):
        """similarity in [0,1]; higher is better. protos (C,W), X (n,W)."""
        out = np.empty((X.shape[0], protos.shape[0]), dtype=np.float32)
        for i in range(X.shape[0]):
            out[i] = 1.0 - popcount_rows(protos ^ X[i]) / self.D
        return out

    # training state = integer accumulators (never shipped to the device)
    def init_acc(self, X, y, C):
        B = unpack_bipolar(X).astype(np.int32)
        acc = np.zeros((C, self.D), dtype=np.int32)
        np.add.at(acc, y, B)
        return acc, B

    def finalize(self, acc):
        return pack_bits((acc > 0).astype(np.uint8))

    def update(self, acc, B, y, pred, wrong):
        np.add.at(acc, y[wrong], B[wrong])
        np.add.at(acc, pred[wrong], -B[wrong])

    def corrupt(self, X, p, rng):
        """Replace a fraction p of components with random values (flip prob p/2)."""
        flip = rng.random((X.shape[0], self.D)) < (p / 2)
        return X ^ pack_bits(flip.astype(np.uint8))

    def model_bytes(self, C):
        return C * self.D // 8


# --------------------------------------------------------------------------
# 3. Phase HDC (discrete FHRR)
# --------------------------------------------------------------------------
class PhaseHDC:
    """Elements are phases in Z_Q. bind = add mod Q, bundle = phasor sum, sim = mean cos."""
    name = "PhaseHDC"

    def __init__(self, bits_budget=8192, Q=16, salt=0):
        self.Q = Q
        self.bits_per = int(np.log2(Q))
        self.D = bits_budget // self.bits_per
        assert (self.D * 1) % 8 == 0
        self.W = self.D // 8  # words needed to hold D bytes
        self.im = ProceduralItemMemory(self.W, "phase", salt=salt)
        ang = 2 * np.pi * np.arange(Q) / Q
        self.cos_t = np.cos(ang).astype(np.float32)
        self.sin_t = np.sin(ang).astype(np.float32)
        self.ph = (self.cos_t + 1j * self.sin_t).astype(np.complex64)
        self.mask = Q - 1

    def encode(self, text):
        toks, bigrams = text_features(text)
        rows = [self.im.get(t) for t in toks]
        rows += [(np.roll(self.im.get(a), 1) + self.im.get(b)) & self.mask for a, b in bigrams]
        if not rows:
            return np.zeros(self.D, dtype=np.uint8)
        mat = np.stack(rows)
        z = self.cos_t[mat].sum(0) + 1j * self.sin_t[mat].sum(0)
        return self.quantize(z)

    def quantize(self, z):
        q = np.rint(np.angle(z) / (2 * np.pi / self.Q)).astype(np.int64) % self.Q
        return q.astype(np.uint8)

    def encode_many(self, texts):
        return np.stack([self.encode(t) for t in texts])

    def scores(self, protos, X):
        P = self.ph[protos]            # (C,D)
        Z = self.ph[X]                 # (n,D)
        return (Z @ P.conj().T).real / self.D

    def scores_lut(self, protos, x):
        """Integer-index LUT path for a single query (what an edge device would run)."""
        delta = (protos - x[None, :]) & self.mask      # uint8 wraps mod 256, 256 % Q == 0
        return self.cos_t[delta].sum(axis=1) / self.D

    def init_acc(self, X, y, C):
        Zs = self.ph[X]
        acc = np.zeros((C, self.D), dtype=np.complex64)
        np.add.at(acc, y, Zs)
        return acc, Zs

    def finalize(self, acc):
        return self.quantize(acc)

    def update(self, acc, Zs, y, pred, wrong):
        np.add.at(acc, y[wrong], Zs[wrong])
        np.add.at(acc, pred[wrong], -Zs[wrong])

    def corrupt(self, X, p, rng):
        m = rng.random(X.shape) < p
        r = rng.integers(0, self.Q, X.shape, dtype=np.uint8)
        return np.where(m, r, X)

    def model_bytes(self, C):
        return C * self.D * self.bits_per // 8


# --------------------------------------------------------------------------
# Classifier wrapper (works for both encoders)
# --------------------------------------------------------------------------
class HDClassifier:
    """Prototype classifier. epochs>0 enables perceptron-style retraining; margin>0 additionally
    updates on correctly-classified samples whose win over the best rival is smaller than `margin`
    (in similarity units, 0..1). Margin training makes prototypes discriminative, which matters
    when features are confounded (e.g. an object word that only ever appears with certain actions).
    NOTE: `margin` is an absolute similarity gap. If part of the code carries no signal (diluting similarities), a margin tuned
    for another encoder can become unreachable and training will thrash; re-tune it, or compare against margin=0."""

    def __init__(self, enc, epochs=0, margin=0.0):
        self.enc, self.epochs, self.margin = enc, epochs, margin

    def fit(self, Xenc, y, C):
        self.C = C
        acc, B = self.enc.init_acc(Xenc, y, C)
        self.protos = self.enc.finalize(acc)
        rows = np.arange(len(y))
        for _ in range(self.epochs):
            S = self.enc.scores(self.protos, Xenc)
            pred = S.argmax(1)
            true_s = S[rows, y]
            S2 = S.copy(); S2[rows, y] = -np.inf
            rival = S2.argmax(1)
            viol = (pred != y) | (true_s - S2[rows, rival] < self.margin)
            if not viol.any():
                break
            self.enc.update(acc, B, y, rival, viol)       # push toward truth, away from best rival
            self.protos = self.enc.finalize(acc)
        return self

    def predict(self, Xenc):
        return self.enc.scores(self.protos, Xenc).argmax(1)


# --------------------------------------------------------------------------
# 4. Event-driven cascade search (binary HDC)
# --------------------------------------------------------------------------
class SplitStore:
    """Packed vectors stored as a contiguous prefix block + contiguous rest block."""

    def __init__(self, P, w1=16):
        self.N, self.W, self.w1 = P.shape[0], P.shape[1], w1
        self.pre = np.ascontiguousarray(P[:, :w1])
        self.rest = np.ascontiguousarray(P[:, w1:])
        self.full = P

    def dense(self, q):
        h = popcount_rows(self.full ^ q)
        return int(h.argmin()), self.N * self.W, 1.0

    def cascade(self, q, margin_bits=None, z=2.0):
        """Stage 1: cheap prefix pulse to every node. Stage 2: only nodes within the margin of
        the best prefix score 'fire' and receive the full comparison.
        margin_bits=None -> adaptive: margin = z * sigma, sigma = sqrt(n p (1-p)) with the noise
        rate p estimated from the best prefix distance itself (no tuning per store)."""
        h1 = popcount_rows(self.pre ^ q[: self.w1])
        mn = int(h1.min())
        if margin_bits is None:
            n = self.w1 * 64
            p = mn / n
            margin_bits = int(np.ceil(z * np.sqrt(n * p * (1 - p))))
        active = np.flatnonzero(h1 <= mn + margin_bits)
        h2 = popcount_rows(self.rest[active] ^ q[self.w1:])
        best = int(active[np.argmin(h1[active] + h2)])
        ops = self.N * self.w1 + len(active) * (self.W - self.w1)
        return best, ops, len(active) / self.N
