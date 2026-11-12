"""
run_knowledge.py - Phase D tests: multi-hop symbolic association on a synthetic knowledge graph,
using the sharded holographic memory in src/knowledge.py.  Writes results/knowledge.md.
"""
import os, sys, math, random, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from knowledge import KnowledgeMemory

E, R, FACTS_PER_SUBJECT = 5000, 16, 4
ents = [f"ent{i}" for i in range(E)]
rels = [f"rel{i}" for i in range(R)]

def make_graph(seed=0):
    rng = random.Random(seed)
    g = {}
    for s in ents:
        for r in rng.sample(rels, FACTS_PER_SUBJECT):
            o = rng.choice(ents)
            while o == s: o = rng.choice(ents)
            g[(s, r)] = o
    return g

def build(g, D, f_target):
    F = len(g); S = max(1, round(F / f_target))
    km = KnowledgeMemory(ents, S, D)
    for (s, r), o in g.items(): km.add(s, r, o)
    km.finalize()
    return km

def acc1(km, g, n=1500, seed=1, **kw):
    rng = random.Random(seed); keys = rng.sample(list(g), n)
    return float(np.mean([km.query(s, r, **kw) == g[(s, r)] for s, r in keys]))

def make_paths(g, hops, n, seed=2):
    rng = random.Random(seed); by = {}
    for (s, r), o in g.items(): by.setdefault(s, []).append((r, o))
    out = []
    while len(out) < n:
        s = rng.choice(ents); cur = s; rs = []
        for _ in range(hops):
            r, o = rng.choice(by[cur]); rs.append(r); cur = o
        out.append((s, rs, cur))
    return out

def acc_hops(km, paths, **kw):
    return float(np.mean([km.hops(s, rs, **kw) == t for s, rs, t in paths]))

L = []; A = L.append
g = make_graph(); F = len(g)
EXPLICIT_BITS = math.ceil(math.log2(R)) + math.ceil(math.log2(E))   # relation id + object id per fact (lower bound, no index)
A("# Phase D: sharded holographic knowledge memory\n")
A(f"Synthetic graph: {E} entities, {R} relations, {F:,} facts ({FACTS_PER_SUBJECT} per subject, functional relations). Queries: 1,500 per cell; multi-hop: 1,000 paths. Dense cleanup unless stated.\n")
A(f"An explicit table needs at least {EXPLICIT_BITS} bits/fact (relation id + object id, ignoring indexing). Holographic cost is D / (facts per shard).\n")

A("## K1. 1-hop accuracy vs shard load and dimension\n")
A("| D (bits) | facts/shard (target) | shards | bits/fact | persistent bytes | 1-hop accuracy |")
A("|---|---|---|---|---|---|")
for D in (4096, 8192, 16384):
    for f in (16, 32, 64, 128, 256):
        km = build(g, D, f)
        A(f"| {D} | {f} | {km.S} | {D * km.S / F:.0f} | {km.persistent_bytes():,} | {acc1(km, g)*100:.1f}% |")
        print("K1", D, f, flush=True)

A("\n## K2. Multi-hop accuracy (error compounds per hop)\n")
A("| D | facts/shard | 1 hop | 2 hops | 3 hops | 4 hops |")
A("|---|---|---|---|---|---|")
for D, f in ((8192, 32), (8192, 64), (16384, 64)):
    km = build(g, D, f)
    row = [acc_hops(km, make_paths(g, h, 1000)) for h in (1, 2, 3, 4)]
    A(f"| {D} | {f} | " + " | ".join(f"{x*100:.1f}%" for x in row) + " |")
    print("K2", D, f, flush=True)

A("\n## K3. Corruption tolerance (fraction of stored shard bits flipped), D=8192, 32 facts/shard\n")
A("| bits flipped | 1-hop | 2-hop |")
A("|---|---|---|")
rng = np.random.default_rng(0)
for p in (0.0, 0.05, 0.10, 0.20, 0.30, 0.40):
    km = build(g, 8192, 32); km.corrupt(p, rng)
    A(f"| {p*100:.0f}% | {acc1(km, g)*100:.1f}% | {acc_hops(km, make_paths(g, 2, 1000))*100:.1f}% |")
    print("K3", p, flush=True)

A("\n## K4. Event-driven cleanup (cascade over the 5,000-entity item memory), D=8192, 32 facts/shard\n")
A("Dense cleanup compares the query against all entities; the cascade first compares a prefix and only fully compares entities that 'fire'.\n")
A("| mode | prefix bits | 1-hop accuracy | entity-compare ops vs dense | ms/query |")
A("|---|---|---|---|---|")
for pw, z in ((None, None), (32, 2.0), (32, 3.0), (64, 2.0), (64, 3.0)):
    km = KnowledgeMemory(ents, round(F / 32), 8192, prefix_words=pw or 32)
    for (s, r), o in g.items(): km.add(s, r, o)
    km.finalize()
    keys = random.Random(5).sample(list(g), 1000)
    t0 = time.perf_counter()
    if pw is None:
        a = np.mean([km.query(s, r) == g[(s, r)] for s, r in keys]); ops = 1.0; lab = "dense"; pb = "-"
    else:
        a = np.mean([km.query(s, r, use_cascade=True, z=z) == g[(s, r)] for s, r in keys])
        ops = km.ops_cascade / (len(keys) * E * km.W); lab = f"cascade z={z}"; pb = pw * 64
    ms = (time.perf_counter() - t0) / len(keys) * 1e3
    A(f"| {lab} | {pb} | {a*100:.1f}% | {ops:.2f} | {ms:.2f} |")
    print("K4", lab, pb, flush=True)

os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"), exist_ok=True)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "knowledge.md")
open(out, "w").write("\n".join(L)); print("\n".join(L))
