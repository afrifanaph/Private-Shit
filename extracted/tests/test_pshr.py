"""Sanity tests for the PS-HR reference implementation. Run: python -m unittest discover -s tests -v"""
import os, sys, unittest
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from pshr import (BinaryHDC, PhaseHDC, HDClassifier, SplitStore, ProceduralItemMemory,
                  expand_seed, seed_of, popcount_rows, pack_bits, unpack_bipolar)


class TestProcedural(unittest.TestCase):
    def test_seed_is_deterministic(self):
        a = expand_seed(seed_of("hello"), 128); b = expand_seed(seed_of("hello"), 128)
        self.assertTrue(np.array_equal(a, b))

    def test_distinct_tokens_are_quasi_orthogonal(self):
        a = expand_seed(seed_of("alpha"), 128); b = expand_seed(seed_of("beta"), 128)
        frac = popcount_rows(a ^ b) / 8192
        self.assertTrue(0.46 < frac < 0.54, frac)

    def test_cache_matches_regeneration(self):
        im = ProceduralItemMemory(128)
        self.assertTrue(np.array_equal(im.get("x"), im.regenerate("x")))
        self.assertEqual(im.misses, 1)
        im.get("x"); self.assertEqual(im.hits, 1)


class TestBinaryHDC(unittest.TestCase):
    def test_pack_unpack_roundtrip(self):
        bits = (np.random.default_rng(0).random((3, 8192)) < 0.5).astype(np.uint8)
        self.assertTrue(np.array_equal((unpack_bipolar(pack_bits(bits)) > 0).astype(np.uint8), bits))

    def test_xor_bind_is_self_inverse(self):
        a = expand_seed(1, 128); b = expand_seed(2, 128)
        self.assertTrue(np.array_equal((a ^ b) ^ b, a))

    def test_similar_text_closer_than_unrelated(self):
        enc = BinaryHDC(8192)
        q = enc.encode("please cancel my card")
        near = enc.encode("please cancle my card")          # typo
        far = enc.encode("what is the weather in paris")
        self.assertLess(popcount_rows(q ^ near), popcount_rows(q ^ far))

    def test_encoding_is_deterministic(self):
        e1, e2 = BinaryHDC(8192), BinaryHDC(8192)
        self.assertTrue(np.array_equal(e1.encode("same text"), e2.encode("same text")))


class TestPhaseHDC(unittest.TestCase):
    def test_lut_scores_match_matmul_scores(self):
        enc = PhaseHDC(8192, 16)
        texts = ["cancel my card", "reset password", "book a flight", "order status please"]
        X = enc.encode_many(texts)
        s_mm = enc.scores(X, X[:1])[0]
        s_lut = enc.scores_lut(X, X[0])
        self.assertTrue(np.allclose(s_mm, s_lut, atol=1e-3))
        self.assertEqual(int(np.argmax(s_lut)), 0)


class TestClassifier(unittest.TestCase):
    def test_learns_separable_classes(self):
        tr = [(0, "reset my password please"), (0, "password reset help"), (0, "forgot password reset"),
              (1, "book a flight to rome"), (1, "flight booking to paris"), (1, "need a flight booked")]
        y = np.array([c for c, _ in tr]); txt = [t for _, t in tr]
        for enc in (BinaryHDC(4096), PhaseHDC(4096, 16)):
            clf = HDClassifier(enc, 5).fit(enc.encode_many(txt), y, 2)
            pred = clf.predict(enc.encode_many(["i forgot my password", "flight to london"]))
            self.assertEqual(list(pred), [0, 1], type(enc).__name__)


class TestCascade(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.N, self.W = 5000, 128
        self.P = rng.integers(0, 2 ** 63, (self.N, self.W), dtype=np.uint64) ^ \
                 (rng.integers(0, 2 ** 63, (self.N, self.W), dtype=np.uint64) << np.uint64(1))
        self.rng = rng

    def test_adaptive_cascade_finds_planted_item_with_fewer_ops(self):
        store = SplitStore(self.P, 16)
        idx = self.rng.integers(0, self.N, 30)
        Q = self.P[idx] ^ pack_bits((self.rng.random((30, self.W * 64)) < 0.2).astype(np.uint8))
        res = [store.cascade(q) for q in Q]
        self.assertTrue(all(r[0] == i for r, i in zip(res, idx)))
        self.assertLess(np.mean([r[1] for r in res]) / (self.N * self.W), 0.2)

    def test_cascade_agrees_with_dense_when_margin_is_huge(self):
        store = SplitStore(self.P, 16)
        q = self.P[7] ^ pack_bits((self.rng.random(self.W * 64) < 0.3).astype(np.uint8))
        self.assertEqual(store.cascade(q, margin_bits=10 ** 6)[0], store.dense(q)[0])


class TestKnowledge(unittest.TestCase):
    def test_one_and_two_hop_queries(self):
        from knowledge import KnowledgeMemory
        ents = [f"e{i}" for i in range(300)]
        rng = np.random.default_rng(0)
        g = {(s, f"r{k}"): ents[int(rng.integers(0, 300))] for s in ents for k in range(3)}
        km = KnowledgeMemory(ents, n_shards=20, D=4096)       # ~45 facts per shard
        for (s, r), o in g.items():
            km.add(s, r, o)
        km.finalize()
        keys = list(g)[:200]
        self.assertGreater(np.mean([km.query(s, r) == g[(s, r)] for s, r in keys]), 0.95)
        s0 = "e0"; r1 = "r0"; mid = g[(s0, r1)]; r2 = "r1"
        self.assertEqual(km.hops(s0, [r1, r2]), g[(mid, r2)])

    def test_cascade_cleanup_matches_dense(self):
        from knowledge import KnowledgeMemory
        ents = [f"e{i}" for i in range(200)]
        g = {(s, "r"): ents[(i * 7 + 3) % 200] for i, s in enumerate(ents)}
        km = KnowledgeMemory(ents, n_shards=10, D=8192, prefix_words=32)
        for (s, r), o in g.items():
            km.add(s, r, o)
        km.finalize()
        agree = np.mean([km.query(s, r) == km.query(s, r, use_cascade=True, z=3.0) for s, r in list(g)[:100]])
        self.assertGreater(agree, 0.97)


class TestQA(unittest.TestCase):
    def test_read_store_answer_roundtrip(self):
        from qa import TextQA
        ents = ["Ama", "Kofi", "Accra", "Kumasi", "Ghana", "Zzx", "Zzy"]
        qa = TextQA(ents, D=4096, epochs=30)
        qa.fit([("lives_in", "Zzx lives in Zzy"), ("lives_in", "Zzx resides in Zzy"),
                ("located_in", "Zzx is a city in Zzy"), ("located_in", "Zzx lies in Zzy")],
               [(("lives_in",), "Where does Zzx live?"), (("lives_in",), "In which city does Zzx live?"),
                (("located_in",), "Which country is Zzx in?"), (("located_in",), "In which country is Zzx?"),
                (("lives_in", "located_in"), "In which country does Zzx live?"),
                (("lives_in", "located_in"), "Which country does Zzx live in?")])
        qa.ingest(["Ama lives in Accra", "Kofi resides in Kumasi", "Accra is a city in Ghana", "Kumasi lies in Ghana"])
        qa.build_memory()
        self.assertEqual(qa.ask("Where does Ama live?")[0], "Accra")
        self.assertEqual(qa.ask("In which country does Kofi live?")[0], "Ghana")

    def test_entity_masking(self):
        from qa import TextQA
        qa = TextQA(["Ama", "Accra"], D=4096)
        masked, found = qa.spot("Ama's friend lives in Accra")
        self.assertEqual(found, ["Ama", "Accra"])
        self.assertIn("ent's", masked)


class TestEmbeddingHDC(unittest.TestCase):
    def test_hamming_tracks_angle(self):
        from embed_hdc import EmbeddingHDC
        rng = np.random.default_rng(0); dim = 64
        x = rng.standard_normal(dim)
        table = {}
        for c in (0.9, 0.5, 0.0):
            noise = rng.standard_normal(dim); noise -= noise @ x / (x @ x) * x
            y = c * x / np.linalg.norm(x) + np.sqrt(1 - c * c) * noise / np.linalg.norm(noise)
            table[f"y{c}"] = y
        table["x"] = x / np.linalg.norm(x)
        enc = EmbeddingHDC(lambda ts: np.stack([table[t] for t in ts]), dim, D=8192)
        code = enc.encode_many(["x", "y0.9", "y0.5", "y0.0"])
        frac = [popcount_rows(code[0] ^ code[i]) / 8192 for i in (1, 2, 3)]
        for f, c in zip(frac, (0.9, 0.5, 0.0)):
            self.assertAlmostEqual(f, np.arccos(c) / np.pi, delta=0.03)

    def test_hybrid_shape_and_classifier(self):
        from embed_hdc import EmbeddingHDC, HybridEncoder
        emb = EmbeddingHDC(lambda ts: np.stack([np.eye(8)[len(t) % 8] for t in ts]), 8, D=1024)
        hy = HybridEncoder(BinaryHDC(2048), emb, r=2)
        X = hy.encode_many(["delete file", "create folder"])
        self.assertEqual(X.shape, (2, (2048 + 2 * 1024) // 64))
        y = np.array([0, 1])
        clf = HDClassifier(hy, 5).fit(X, y, 2)
        self.assertEqual(list(clf.predict(X)), [0, 1])


if __name__ == "__main__":
    unittest.main()
