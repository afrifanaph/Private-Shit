"""
run_qa.py - end-to-end text QA demo/evaluation (src/qa.py) on a synthetic world.
Classifiers see only 3 phrasings per relation / question type; evaluation uses phrasings NEVER seen in training.
Writes results/qa.md.
"""
import os, sys, random
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from qa import TextQA

S = {  # relation: (training phrasings, held-out phrasings)   {s}=subject {o}=object
 "lives_in":   (["{s} lives in {o}", "{s} resides in {o}", "{s} is based in {o}"],
                ["{s} stays in {o}", "{s} has a home in {o}", "{s} is living in {o}", "{s} dwells in {o}"]),
 "born_in":    (["{s} was born in {o}", "{s} is a native of {o}", "{s} comes from {o}"],
                ["{s} is originally from {o}", "{s} hails from {o}", "{s} was born and raised in {o}"]),
 "works_at":   (["{s} works at {o}", "{s} is employed by {o}", "{s} has a job at {o}"],
                ["{s} works for {o}", "{s} is on the staff of {o}", "{s} is hired by {o}"]),
 "founded":    (["{s} founded {o}", "{s} is the founder of {o}", "{s} started the company {o}"],
                ["{s} established {o}", "{s} set up {o}", "{s} created {o}"]),
 "located_in": (["{s} is a city in {o}", "{s} lies in {o}", "{s} is located in {o}"],
                ["{s} can be found in {o}", "{s} is situated in {o}", "{s} sits in {o}"]),
 "best_friend":(["{s} is best friends with {o}", "{s} is close to {o}", "{s} trusts {o} most"],
                ["{s} hangs out with {o}", "{s} is tight with {o}", "{s} is buddies with {o}"]),
}
Q = {  # relation chain: (training phrasings, held-out phrasings)
 ("lives_in",):   (["Where does {s} live?", "In which city does {s} live?", "Which city does {s} reside in?"],
                   ["What city is {s} living in?", "Which city does {s} stay in?", "Where is {s} based?"]),
 ("born_in",):    (["Where was {s} born?", "In which city was {s} born?", "Which city is {s} a native of?"],
                   ["What is the birthplace of {s}?", "Which city does {s} come from?", "Where is {s} originally from?"]),
 ("works_at",):   (["Where does {s} work?", "Which company employs {s}?", "Who does {s} work at?"],
                   ["Which company does {s} work for?", "What is the employer of {s}?", "Who is {s} employed by?"]),
 ("founded",):    (["What did {s} found?", "Which company did {s} found?", "What company was founded by {s}?"],
                   ["What did {s} establish?", "Which firm did {s} set up?", "What did {s} start?"]),
 ("located_in",): (["In which country is {s}?", "Which country is {s} in?", "What country is the city {s} in?"],
                   ["What country does {s} belong to?", "{s} is located in which country?", "Where is the city {s} located?"]),
 ("best_friend",):(["Who is {s} best friends with?", "Who is the best friend of {s}?", "Who is {s} close to?"],
                   ["Who is {s} buddies with?", "Name the best pal of {s}", "Who does {s} hang out with?"]),
 ("lives_in", "located_in"): (["In which country does {s} live?", "Which country does {s} live in?", "What country does {s} reside in?"],
                   ["What country is {s} living in?", "{s} lives in which country?", "In what country does {s} stay?"]),
 ("born_in", "located_in"):  (["In which country was {s} born?", "Which country was {s} born in?", "What country is {s} a native of?"],
                   ["What country is {s} originally from?", "{s} was born in which country?", "In what country was {s} brought into the world?"]),
 ("best_friend", "lives_in"):(["Where does the best friend of {s} live?", "In which city does {s}'s best friend live?", "Which city does the best friend of {s} reside in?"],
                   ["What city is the best friend of {s} living in?", "Which city does {s}'s best friend stay in?", "Where is the best friend of {s} based?"]),
}

def names(n, rng, used, cap=True):
    syl = ["ka", "ko", "fi", "am", "ba", "ne", "ti", "su", "ra", "lo", "mi", "da", "ye", "zu", "po", "ge"]
    out = []
    while len(out) < n:
        w = "".join(rng.choice(syl) for _ in range(rng.randint(3, 4)))
        w = w.capitalize()
        if w not in used: used.add(w); out.append(w)
    return out

def world(seed=0, n_people=60, n_cities=20, n_countries=6, n_companies=10):
    rng = random.Random(seed); used = set()
    people, cities, countries, companies = (names(k, rng, used) for k in (n_people, n_cities, n_countries, n_companies))
    g = {}
    for c in cities: g[(c, "located_in")] = rng.choice(countries)
    for p in people:
        g[(p, "lives_in")] = rng.choice(cities); g[(p, "born_in")] = rng.choice(cities)
        g[(p, "works_at")] = rng.choice(companies)
        g[(p, "best_friend")] = rng.choice([x for x in people if x != p])
    for co, p in zip(companies, rng.sample(people, len(companies))): g[(p, "founded")] = co
    return people + cities + countries + companies, people, g, rng

def follow(g, s, chain):
    for r in chain: s = g[(s, r)]
    return s

def run(seed, train_phrasing_only=False, extra=0, extra_fixed=False, encoder_factory=None):
    ents, people, g, rng = world(seed)
    # entity placeholders must be real entity names for the spotter: use two reserved dummy entities
    ents2 = ents + ["Zzx", "Zzy"]
    qa = TextQA(ents2, encoder=encoder_factory() if encoder_factory else None)
    pick = random.Random(1000 + seed)       # which held-out phrasings become extra training data (never the last one)
    exS = {r: pick.sample(ho[:-1], extra) for r, (tr, ho) in S.items()}
    exQ = {c: pick.sample(ho[:-1], extra) for c, (tr, ho) in Q.items()}
    qa.fit([(r, t.format(s="Zzx", o="Zzy")) for r, (tr, ho) in S.items() for t in tr + exS[r]],
           [(c, t.format(s="Zzx")) for c, (tr, ho) in Q.items() for t in tr + exQ[c]])
    which = 0 if train_phrasing_only else 1
    # 1) statements written in held-out phrasings
    stmts, truth = [], []
    for (s, r), o in g.items():
        stmts.append(rng.choice(S[r][1][-1:] if which and extra_fixed else S[r][which]).format(s=s, o=o)); truth.append((s, r, o))
    rel_acc = float(np.mean([qa.classify_statement(t)[0] == tr[1] for t, tr in zip(stmts, truth)]))
    skipped = qa.ingest(stmts); qa.build_memory()
    # oracle-extraction memory (ground-truth facts) to separate extraction errors from memory errors
    from knowledge import KnowledgeMemory
    orc = KnowledgeMemory(ents2, qa.km.S, qa.D)
    for (s, r), o in g.items(): orc.add(s, r, o)
    orc.finalize()
    # 2) questions in held-out phrasings
    res = {1: [], 2: []}; qacc = []; orc_res = {1: [], 2: []}; samples = []
    for chain in Q:
        pool = [p for p in people] if chain[0] != "located_in" else [e for e in ents if (e, "located_in") in g]
        if chain == ("founded",): pool = [p for (p, r) in g if r == "founded"]
        for s in rng.sample(pool, min(25, len(pool))):
            text = rng.choice(Q[chain][1][-1:] if which and extra_fixed else Q[chain][which]).format(s=s); gold = follow(g, s, chain)
            ans, pred_chain = qa.ask(text)
            qacc.append(pred_chain == chain); res[len(chain)].append(ans == gold)
            orc_res[len(chain)].append(orc.hops(s, list(chain)) == gold)
            if len(samples) < 14 and rng.random() < 0.25: samples.append((text, ans, gold, pred_chain))
    return dict(rel_acc=rel_acc, skipped=skipped, q_type=float(np.mean(qacc)),
                e2e1=float(np.mean(res[1])), e2e2=float(np.mean(res[2])),
                or1=float(np.mean(orc_res[1])), or2=float(np.mean(orc_res[2])),
                n_facts=len(g), shards=qa.km.S, mem_bytes=qa.km.persistent_bytes()), samples

if __name__ == "__main__":
    def avg(rs, k): return np.mean([r[0][k] for r in rs]) * 100
    L = ["# End-to-end text QA (src/qa.py)\n",
         "World: 60 people, 20 cities, 6 countries, 10 companies, 6 relations, ~290 facts, memory D=8192 with ~32 facts/shard. "
         "Relation and question-type classifiers were trained on **3 phrasings each** and evaluated on **phrasings never seen in training**. 5 random worlds, 25 questions per question type per world.\n"]
    for title, tpo in (("Held-out phrasings (the real test)", False), ("Training phrasings (sanity check)", True)):
        rs = [run(s, tpo) for s in range(5)]
        L += [f"## {title}\n", "| Stage | Accuracy |", "|---|---|",
              f"| Statement -> relation (reading) | {avg(rs,'rel_acc'):.1f}% |",
              f"| Question -> question type (understanding) | {avg(rs,'q_type'):.1f}% |",
              f"| End-to-end, 1-step questions | {avg(rs,'e2e1'):.1f}% |",
              f"| End-to-end, 2-step questions | {avg(rs,'e2e2'):.1f}% |",
              f"| Memory alone with perfect reading (1-step / 2-step) | {avg(rs,'or1'):.1f}% / {avg(rs,'or2'):.1f}% |",
              f"| Memory size | {int(np.mean([r[0]['mem_bytes'] for r in rs])):,} bytes for {rs[0][0]['n_facts']} facts |\n"]
        if not tpo:
            L += ["Example questions (held-out phrasing; answer / correct / detected question type):\n", "```"]
            L += [f"{t:55s} -> {a} (correct: {g_}) {c}" for t, a, g_, c in rs[0][1]] + ["```\n"]
    L += ["## How many example phrasings are needed? (fixed test phrasing, 10 worlds)\n",
          "Each relation / question type starts with 3 training phrasings; `+k` adds k randomly chosen held-out phrasings to training. Every row is tested on the SAME single never-trained phrasing per type (the last held-out one), so rows are comparable; wording variety in the test is small, so treat as indicative.\n",
          "| Training phrasings per type | Reading (statement -> relation) | Question type | End-to-end 1-step | End-to-end 2-step |", "|---|---|---|---|---|"]
    for k in (0, 1, 2):
        rs = [run(s_, False, k, True) for s_ in range(10)]
        L.append(f"| 3 + {k} | {avg(rs,'rel_acc'):.1f}% | {avg(rs,'q_type'):.1f}% | {avg(rs,'e2e1'):.1f}% | {avg(rs,'e2e2'):.1f}% |")
    L.append("")
    os.makedirs("../results", exist_ok=True); open("../results/qa.md", "w").write("\n".join(L)); print("\n".join(L))
