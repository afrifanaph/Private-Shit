# Semantic-encoder experiments (negative results)

Benchmark: 16 hand-written sysadmin intents, 3 plain-wording training utterances each (48), 128 paraphrase test utterances using different words. Author-written, small (1 item = 0.78 points; binomial SE about 3-4 points). Corpus: ~4.4M words of local Python/NumPy/SciPy/scikit-learn docstrings.

```
=== (1) random-indexing semantic encoder ===
vocab coverage of test words: 88%   (n_test=128, 1 item = 0.78 pts, binomial SE ~ 3-4 pts)

 63.3%  Lexical HDC (8192b)
 57.0%  Semantic HDC only (2048b, random indexing)
 64.8%  Hybrid HDC (8192b lexical + 1x2048b semantic)
 65.6%  Hybrid HDC (8192b lexical + 2x2048b semantic)
 63.3%  Hybrid HDC (8192b lexical + 4x2048b semantic)
 62.5%  Hybrid HDC (8192b lexical + 8x2048b semantic)
 63.3%  Hybrid, +retrain 15 ep (r=4)
 64.8%  Float RI centroid (cosine, same embeddings)
 67.2%  Float RI + LogReg
 71.1%  Lexical TF-IDF + LogReg
  6.2%  Chance

=== (2) IDF-weighted lexical bundling (exploratory; swept against the same 128 test items) ===
baseline unweighted lexical: 63.3
idf^1  trigram weight 0.0: 57.8%
idf^1  trigram weight 0.3: 61.7%
idf^1  trigram weight 1.0: 60.9%
idf^2  trigram weight 0.0: 50.0%
idf^2  trigram weight 0.3: 57.8%
idf^2  trigram weight 1.0: 53.1%
idf^3  trigram weight 0.0: 40.6%
idf^3  trigram weight 0.3: 53.1%
idf^3  trigram weight 1.0: 47.7%
idf^2 trig 0.3 floor 0: 57.8%
idf^2 trig 0.3 floor 3: 53.9%
idf^2 trig 0.3 floor 5: 45.3%
```

**Reading:** random indexing from this corpus gives no reliable gain (hybrid 62-66% vs lexical 63%, within noise); binarizing the embeddings costs ~8 points versus their float form; IDF weighting hurts monotonically, plausibly because rare words in paraphrases are mostly unseen in training and add noise, whereas a trained linear model gives unseen features zero weight. The IDF sweep was run against the same 128 test items, so even its best cell should not be read as a tuned result. The likely missing ingredient is data (a pretrained encoder or a conversational corpus), not the binding/bundling machinery.
