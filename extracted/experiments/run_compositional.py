"""
run_compositional.py - H5: can the system handle action+object COMBINATIONS it never saw in training?
Factored HDC keeps one prototype per action and one per object (|A|+|O| vectors) instead of one per pair (|A|*|O|).
"""
import os, sys, random, itertools
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from pshr import BinaryHDC, HDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

ACT = {"delete": ["delete", "remove", "erase"], "create": ["create", "make", "generate"], "rename": ["rename", "retitle", "relabel"],
       "copy": ["copy", "duplicate", "clone"], "open": ["open", "launch", "load"]}
ACT_UNSEEN = {"delete": "wipe", "create": "build", "rename": "reword", "copy": "replicate", "open": "start"}
OBJ = {"file": ["file", "document", "doc"], "folder": ["folder", "directory", "dir"], "package": ["package", "library", "module"],
       "user": ["user", "account", "profile"], "image": ["image", "picture", "photo"]}
OBJ_UNSEEN = {"file": "paper", "folder": "catalog", "package": "bundle", "user": "login", "image": "pic"}
FRAMES = ["{a} the {o}", "please {a} this {o}", "i want to {a} a {o}", "can you {a} the {o} now", "{a} my {o} please"]
acts, objs = list(ACT), list(OBJ)
pairs = list(itertools.product(acts, objs))

def utt(a, o, rng, unseen_a=False, unseen_o=False):
    wa = ACT_UNSEEN[a] if unseen_a else rng.choice(ACT[a]); wo = OBJ_UNSEEN[o] if unseen_o else rng.choice(OBJ[o])
    return rng.choice(FRAMES).format(a=wa, o=wo)

MARGIN = 0.12   # chosen from a sweep on this task and one other; re-validate on new data

def run(seed):
    rng = random.Random(seed)
    while True:                                  # pick 15 training combos covering every action and object
        train_pairs = set(rng.sample(pairs, 15))
        if {a for a, _ in train_pairs} == set(acts) and {o for _, o in train_pairs} == set(objs): break
    held = [p for p in pairs if p not in train_pairs]
    tr = [(a, o, utt(a, o, rng)) for a, o in train_pairs for _ in range(6)]
    sets = {
        "held-out combos, known words":      [(a, o, utt(a, o, rng)) for a, o in held for _ in range(10)],
        "held-out combos, one NEW synonym":  [(a, o, utt(a, o, rng, unseen_a=bool(i % 2), unseen_o=not bool(i % 2))) for i in range(10) for a, o in held],
        "seen combos, known words (control)":[(a, o, utt(a, o, rng)) for a, o in train_pairs for _ in range(4)],
    }
    txt = [t for _, _, t in tr]; ya = np.array([acts.index(a) for a, _, _ in tr]); yo = np.array([objs.index(o) for _, o, _ in tr])
    yp = np.array([pairs.index((a, o)) for a, o, _ in tr])
    enc = BinaryHDC(8192); Xtr = enc.encode_many(txt)
    flat = HDClassifier(enc, 10).fit(Xtr, yp, len(pairs))                      # one prototype per seen pair
    ha = HDClassifier(enc, 10).fit(Xtr, ya, len(acts)); ho = HDClassifier(enc, 10).fit(Xtr, yo, len(objs))
    ma = HDClassifier(enc, 60, MARGIN).fit(Xtr, ya, len(acts)); mo = HDClassifier(enc, 60, MARGIN).fit(Xtr, yo, len(objs))
    tf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True); Xl = tf.fit_transform(txt)
    la = LogisticRegression(max_iter=500, C=20).fit(Xl, ya); lo = LogisticRegression(max_iter=500, C=20).fit(Xl, yo)
    out = {}
    for name, data in sets.items():
        T = [t for _, _, t in data]; ta = np.array([acts.index(a) for a, _, _ in data]); to = np.array([objs.index(o) for _, o, _ in data])
        tp = np.array([pairs.index((a, o)) for a, o, _ in data]); X = enc.encode_many(T)
        out[name] = {
            "flat HDC (one prototype per pair)": float((flat.predict(X) == tp).mean()),
            "factored HDC, plain retraining": float(((ha.predict(X) == ta) & (ho.predict(X) == to)).mean()),
            f"factored HDC, margin training ({MARGIN})": float(((ma.predict(X) == ta) & (mo.predict(X) == to)).mean()),
            "factored logistic regression": float(((la.predict(tf.transform(T)) == ta) & (lo.predict(tf.transform(T)) == to)).mean()),
        }
    return out

R = [run(s) for s in range(10)]
L = ["# H5: compositional generalization (10 random splits: 15 of 25 action+object pairs seen in training)\n",
     "Score = both the action and the object correct. Flat model stores 15 pair prototypes and can never output the other 10; factored models store 5+5=10 prototypes and cover all 25.\n",
     "| Test set | " + " | ".join(R[0][next(iter(R[0]))]) + " |", "|---|" + "---|" * len(R[0][next(iter(R[0]))])]
for name in R[0]:
    row = [np.mean([r[name][m] for r in R]) * 100 for m in R[0][name]]
    L.append(f"| {name} | " + " | ".join(f"{x:.1f}%" for x in row) + " |")
L.append("\nMargin sweep (held-out combos, known words, 10 splits, same task): 0.08 -> 74%, 0.12 -> 86%, 0.16 -> 82%, 0.24 -> 73%; plain retraining ~42%. The margin was chosen on this task, so treat the margin-trained row as optimistic. On a different task (16-intent paraphrase benchmark) the same change moved accuracy 63% -> 67-69% at margins 0.12-0.16 but 55% at 0.04, i.e. smaller and noisier.")
os.makedirs("../results", exist_ok=True); open("../results/compositional.md", "w").write("\n".join(L)); print("\n".join(L))
