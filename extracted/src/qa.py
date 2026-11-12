"""
qa.py - tiny end-to-end question answering on top of the HDC pieces.

  sentence --(spot entities, mask them)--> HDC classifier --> relation   --> KnowledgeMemory.add(s, rel, o)
  question --(spot entity,  mask it)-----> HDC classifier --> relation chain --> KnowledgeMemory.hops(s, chain)

Entities are recognised from a known-name list (case-sensitive). Relation detection and question-type detection
are few-shot HDC classifiers (margin-trained). This is NOT general language understanding: it handles
sentences with exactly two known entities (subject first) and questions with one known entity.
"""
import re
import numpy as np
from pshr import BinaryHDC, HDClassifier
from knowledge import KnowledgeMemory

_TOK = re.compile(r"[A-Za-z][A-Za-z']*")


class TextQA:
    def __init__(self, entities, D=8192, margin=0.12, epochs=60, facts_per_shard=32, encoder=None):
        """encoder: any BinaryHDC-compatible text encoder (default lexical BinaryHDC); e.g. a HybridEncoder
        from embed_hdc.py. D is the memory dimension for the knowledge store."""
        self.entities = list(entities)
        self.ent = set(self.entities)
        self.enc = encoder if encoder is not None else BinaryHDC(D)
        self.margin, self.epochs, self.fps, self.D = margin, epochs, facts_per_shard, D
        self.pending = []
        self.km = None

    # -- entity spotting ------------------------------------------------------------
    def spot(self, text):
        found, out = [], text
        for m in _TOK.finditer(text):
            w = m.group(0)
            base = w[:-2] if w.endswith("'s") else w
            if base in self.ent:
                found.append(base)
        masked = _TOK.sub(lambda m: "ent" + ("'s" if m.group(0).endswith("'s") else "")
                          if (m.group(0)[:-2] if m.group(0).endswith("'s") else m.group(0)) in self.ent else m.group(0), text)
        return masked, found

    # -- training -------------------------------------------------------------------
    def _fit_clf(self, examples):
        labels = sorted({l for l, _ in examples}, key=str)
        X = self.enc.encode_many([self.spot(t)[0] for _, t in examples])
        y = np.array([labels.index(l) for l, _ in examples])
        return HDClassifier(self.enc, self.epochs, self.margin).fit(X, y, len(labels)), labels

    def fit(self, statement_examples, question_examples):
        if hasattr(self.enc, "fit_center") or hasattr(getattr(self.enc, "emb", None), "fit_center"):
            unl = [self.spot(t)[0] for _, t in statement_examples + question_examples]     # unlabeled text only
            (self.enc if hasattr(self.enc, "fit_center") else self.enc.emb).fit_center(unl)
        self.rel_clf, self.rel_labels = self._fit_clf(statement_examples)
        self.q_clf, self.q_labels = self._fit_clf(question_examples)
        return self

    # -- reading --------------------------------------------------------------------
    def classify_statement(self, text):
        m, ents = self.spot(text)
        return self.rel_labels[int(self.rel_clf.predict(self.enc.encode_many([m]))[0])], ents

    def ingest(self, sentences):
        skipped = 0
        for s in sentences:
            rel, ents = self.classify_statement(s)
            if len(ents) == 2:
                self.pending.append((ents[0], rel, ents[1]))
            else:
                skipped += 1
        return skipped

    def build_memory(self):
        shards = max(1, round(len(self.pending) / self.fps))
        self.km = KnowledgeMemory(self.entities, shards, self.D)
        for s, r, o in self.pending:
            self.km.add(s, r, o)
        self.km.finalize()

    # -- answering ------------------------------------------------------------------
    def classify_question(self, text):
        m, ents = self.spot(text)
        return self.q_labels[int(self.q_clf.predict(self.enc.encode_many([m]))[0])], ents

    def ask(self, question):
        chain, ents = self.classify_question(question)
        if not ents:
            return None, chain
        return self.km.hops(ents[0], list(chain)), chain
