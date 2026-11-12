# Embedding hybrid: plumbing check with a stand-in embedding

Model = the locally trained random-indexing vectors (weak; 4.4M-word docstring corpus). This verifies that `EmbeddingHDC` / `HybridEncoder` work with the classifiers and the QA pipeline. It does **not** answer whether a real pretrained sentence model closes the paraphrase gap: that run (`--model sentence-transformers/all-MiniLM-L6-v2`) could not be executed in the sandbox.

```
=== (a) 16-intent paraphrase benchmark (model: ri; 1 item = 0.78 pts, SE ~3-4 pts) ===
  lexical   r=0  margin=0.00:  63.3%
  lexical   r=0  margin=0.12:  67.2%
  embedding r=0  margin=0.00:  60.9%
  embedding r=0  margin=0.12:  68.0%
  hybrid    r=1  margin=0.00:  64.1%
  hybrid    r=1  margin=0.12:  69.5%
  hybrid    r=2  margin=0.00:  65.6%
  hybrid    r=2  margin=0.12:  71.1%
  hybrid    r=4  margin=0.00:  64.1%
  hybrid    r=4  margin=0.12:  69.5%

=== (b) text QA, unseen phrasings (fixed test phrasing, 5 worlds) ===
  lexical : reading 73.6%  question-type 47.6%  end-to-end 1-step 25.8%  2-step 9.9%
  hybrid  : reading 80.1%  question-type 54.1%  end-to-end 1-step 29.2%  2-step 39.7%
```

Reading: margin training (0.12) gave +4 to +6 points over plain training for every encoder on this benchmark (consistent direction, still within about 1.5 standard errors per cell). Hybrid + margin (71.1%) matches TF-IDF logistic regression (71.1%) here, but 128 test items cannot separate differences of a few points. QA gains on unseen phrasings (e.g. 2-step 9.9% -> 39.7%) are from 5 worlds with one fixed phrasing set and are indicative only.
