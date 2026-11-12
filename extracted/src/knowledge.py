"""
knowledge.py - sharded holographic triple store (Phase D: multi-step symbolic association).

Each fact (subject, relation, object) is stored as the XOR trace  S ^ roll(R) ^ O  bundled (majority)
into ONE shard chosen by hash(subject). A query (s, r) touches a single shard (sparse activation):
    guess = shard ^ S ^ roll(R)   ~  O + crosstalk noise
and is cleaned up by nearest-neighbour search over the entity item memory (optionally via the
event-driven cascade). Multi-hop queries feed each cleaned-up answer into the next hop.

Item vectors are procedural (regenerated from a hash of the name), so persistent storage is the shard
memories plus the list of entity names - not a table of vectors.
"""
import numpy as np
from pshr import (ProceduralItemMemory, SplitStore, seed_of, popcount_rows, pack_bits)


class KnowledgeMemory:
    def __init__(self, entities, n_shards, D=8192, salt=0, prefix_words=32):
        assert D % 64 == 0
        self.D, self.W, self.S = D, D // 64, n_shards
        self.entities = list(entities)
        self.eidx = {e: i for i, e in enumerate(self.entities)}
        self.im = ProceduralItemMemory(self.W, "binary", cache_size=10 ** 7, salt=salt)
        self.counts = np.zeros((n_shards, D), dtype=np.int32)      # training-time accumulators
        self.n_in_shard = np.zeros(n_shards, dtype=np.int32)
        self.bits = None                                           # deployed binary shards (S, W)
        self.tie = np.unpackbits(self.im.get("__tie__").view(np.uint8))
        self._ent_mat = None
        self._store = None
        self.prefix_words = prefix_words
        self.ops_dense = self.ops_cascade = 0

    # -- helpers ----------------------------------------------------------------
    def shard_of(self, subject):
        return seed_of(subject, "shard") % self.S

    def _rel(self, r):
        return np.roll(self.im.get("rel:" + r), 1)

    def _ent(self, e):
        return self.im.get("ent:" + e)

    # -- write ------------------------------------------------------------------
    def add(self, s, r, o):
        trace = self._ent(s) ^ self._rel(r) ^ self._ent(o)
        sh = self.shard_of(s)
        self.counts[sh] += np.unpackbits(trace.view(np.uint8)).astype(np.int32) * 2 - 1
        self.n_in_shard[sh] += 1

    def finalize(self):
        """Binarize shards (majority). Ties broken by a fixed pseudo-random vector."""
        b = np.where(self.counts == 0, self.tie[None, :], (self.counts > 0).astype(np.uint8))
        self.bits = np.ascontiguousarray(pack_bits(b)).reshape(self.S, self.W)
        self._ent_mat = np.stack([self._ent(e) for e in self.entities])
        self._store = SplitStore(self._ent_mat, self.prefix_words)

    # -- read -------------------------------------------------------------------
    def query(self, s, r, use_cascade=False, z=2.0):
        q = self.bits[self.shard_of(s)] ^ self._ent(s) ^ self._rel(r)
        if use_cascade:
            i, ops, _ = self._store.cascade(q, None, z)
            self.ops_cascade += ops
        else:
            i, ops, _ = self._store.dense(q)
            self.ops_dense += ops
        return self.entities[i]

    def hops(self, s, rels, **kw):
        cur = s
        for r in rels:
            cur = self.query(cur, r, **kw)
        return cur

    # -- accounting ---------------------------------------------------------------
    def persistent_bytes(self):
        """Shard memories + entity name strings (item vectors are regenerated from names)."""
        return self.S * self.D // 8 + sum(len(e) + 1 for e in self.entities)

    def runtime_ram_bytes(self):
        return self.persistent_bytes() + self._ent_mat.nbytes

    def corrupt(self, p, rng):
        """Flip a fraction p of the stored shard bits (models approximate/faulty memory)."""
        flips = pack_bits((rng.random((self.S, self.D)) < p).astype(np.uint8)).reshape(self.S, self.W)
        self.bits = self.bits ^ flips
