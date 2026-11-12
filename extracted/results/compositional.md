# H5: compositional generalization (10 random splits: 15 of 25 action+object pairs seen in training)

Score = both the action and the object correct. Flat model stores 15 pair prototypes and can never output the other 10; factored models store 5+5=10 prototypes and cover all 25.

| Test set | flat HDC (one prototype per pair) | factored HDC, plain retraining | factored HDC, margin training (0.12) | factored logistic regression |
|---|---|---|---|---|
| held-out combos, known words | 0.0% | 45.2% | 89.2% | 94.6% |
| held-out combos, one NEW synonym | 0.0% | 6.8% | 13.8% | 10.6% |
| seen combos, known words (control) | 51.0% | 80.3% | 98.2% | 98.7% |

Margin sweep (held-out combos, known words, 10 splits, same task): 0.08 -> 74%, 0.12 -> 86%, 0.16 -> 82%, 0.24 -> 73%; plain retraining ~42%. The margin was chosen on this task, so treat the margin-trained row as optimistic. On a different task (16-intent paraphrase benchmark) the same change moved accuracy 63% -> 67-69% at margins 0.12-0.16 but 55% at 0.04, i.e. smaller and noisier.